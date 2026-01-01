"""
Тренер для інверсійно-трансформаторного ядра.
"""
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.adaptive_controller import AdaptiveInversionController
from core.inversion_transformer import InversionTransformerCore
from core.perturbations import get_perturbation


@dataclass
class TrainConfig:
    """
    Конфігурація для тренування моделі.

    Параметри даних:
        data_mode (str): Режим генерації даних ('finance', 'sine', 'anomalies')
        n_samples (int): Кількість зразків
        seq_len (int): Довжина послідовності
        input_dim (int): Розмірність вхідних ознак
        train_split (float): Частка даних для тренування (default: 0.8)

    Параметри моделі:
        d_model (int): Розмірність моделі (default: 64)
        num_layers (int): Кількість transformer шарів (default: 2)
        num_heads (int): Кількість attention heads (default: 4)
        d_ff (int): Розмірність feedforward мережі (default: 256)
        dropout (float): Dropout rate (default: 0.1)
        output_dim (int): Розмірність виходу (default: 1)
        use_representation_for_inv (bool): Використовувати репрезентації для інверсії (default: False)

    Параметри навчання:
        batch_size (int): Розмір батча (default: 32)
        num_epochs (int): Кількість епох (default: 50)
        learning_rate (float): Learning rate (default: 1e-3)
        weight_decay (float): Weight decay для оптимізатора (default: 1e-5)

    Параметри інверсії:
        use_inversion (bool): Використовувати інверсійну регуляризацію (default: False)
        inv_weight (float): Вага інверсійної втрати (default: 0.1)
        perturbation_mode (str): Режим збурень ('gaussian', 'timestep_dropout') (default: 'gaussian')
        perturbation_std (float): Std для gaussian noise (default: 0.01)

    Інші параметри:
        seed (int): Random seed (default: 42)
        results_root (str): Коренева директорія для результатів (default: 'results')
        device (str): Пристрій ('cuda', 'cpu', 'auto') (default: 'auto')
    """

    # Параметри даних
    data_mode: str = "finance"
    n_samples: int = 1000
    seq_len: int = 50
    input_dim: int = 1
    train_split: float = 0.8

    # Параметри моделі
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    output_dim: int = 1
    use_representation_for_inv: bool = False

    # Параметри навчання
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Параметри інверсії
    use_inversion: bool = False
    inv_weight: float = 0.1
    perturbation_mode: str = "gaussian"
    perturbation_std: float = 0.01

    # Параметри адаптивної інверсії
    adaptive_mode: str = "off"  # {'off', 'static', 'online'}
    adaptive_warmup: int = 5  # Кількість епох warmup перед online адаптацією
    adaptive_step: int = 3  # Крок адаптації (кожні N епох)
    adaptive_eta: float = 0.2  # Learning rate для адаптації inv_weight
    adaptive_eta2: float = 0.1  # Learning rate для val_loss gradient
    adaptive_r_target: float = 0.15  # Цільове співвідношення inv_loss/base_loss
    inv_weight_min: float = 0.0  # Мінімальний inv_weight
    inv_weight_max: float = 1.2  # Максимальний inv_weight
    trend_correction_alpha: float = 0.35  # Коефіцієнт корекції тренду
    deterministic_flat_thr: float = 0.15  # Поріг spectral_flatness
    deterministic_sent_thr: float = 0.3  # Поріг sample_entropy
    deterministic_guard_mode: str = "soft"  # {'soft', 'hard'}
    ema_alpha: float = 0.9  # EMA згладжування для loss tracking

    # Інші параметри
    seed: int = 42
    results_root: str = "results"
    device: str = "auto"


class InversionTrainer:
    """
    Тренер для інверсійно-трансформаторного ядра.
    """

    def __init__(self, cfg: TrainConfig):
        """
        Ініціалізація тренера.

        Args:
            cfg (TrainConfig): Конфігурація тренування
        """
        self.cfg = cfg

        # Встановити random seed
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Визначити пристрій
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        print(f"Використовується пристрій: {self.device}")

        # Ініціалізація моделі
        self.model = InversionTransformerCore(
            input_dim=cfg.input_dim,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            output_dim=cfg.output_dim,
            use_representation_for_inv=cfg.use_representation_for_inv,
        ).to(self.device)

        # Ініціалізація оптимізатора
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # Ініціалізація критерію втрати
        self.criterion = nn.MSELoss()

        # Ініціалізація стратегії збурень
        if cfg.use_inversion:
            self.perturbation = get_perturbation(cfg.perturbation_mode, std=cfg.perturbation_std)
        else:
            self.perturbation = None

        # Ініціалізація адаптивного контролера
        if cfg.adaptive_mode in ["static", "online"]:
            self.adaptive_controller = AdaptiveInversionController(
                inv_weight_min=cfg.inv_weight_min,
                inv_weight_max=cfg.inv_weight_max,
                trend_correction_alpha=cfg.trend_correction_alpha,
                deterministic_flat_thr=cfg.deterministic_flat_thr,
                deterministic_sent_thr=cfg.deterministic_sent_thr,
                deterministic_guard_mode=cfg.deterministic_guard_mode,
                verbose=True,
            )
        else:
            self.adaptive_controller = None

    def train_once(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        callback: Optional[Callable] = None,
        tag: str = "default",
    ) -> dict:
        """
        Повний цикл тренування з логуванням.

        Args:
            X_train (np.ndarray): Тренувальні дані (n_samples, seq_len, input_dim)
            Y_train (np.ndarray): Тренувальні таргети (n_samples,)
            X_val (np.ndarray): Валідаційні дані
            Y_val (np.ndarray): Валідаційні таргети
            callback (Optional[Callable]): Callback функція для кожної епохи
            tag (str): Тег для run (default: 'default')

        Returns:
            Dict: Словник з результатами тренування
        """
        # Створити директорію для результатів
        run_dir = self._generate_run_dir(tag)
        os.makedirs(run_dir, exist_ok=True)

        print(f"Результати будуть збережені в: {run_dir}")

        # Конвертувати в torch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        Y_train_t = torch.FloatTensor(Y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val).unsqueeze(1).to(self.device)

        # Створити DataLoader
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # Відкрити файл для логування метрик по епохах
        metrics_file = os.path.join(run_dir, "epoch_metrics.jsonl")

        best_val_loss = float("inf")
        train_history = []

        # Адаптивна інверсія: статична рекомендація
        if self.cfg.adaptive_mode == "static" and self.adaptive_controller is not None:
            print("\n" + "=" * 70)
            print("🔍 ADAPTIVE INVERSION: Static recommendation")
            print("=" * 70)
            recommended_weight, explanation = self.adaptive_controller.recommend(X_train, Y_train)
            self.cfg.inv_weight = recommended_weight
            print(f"Рекомендований inv_weight: {recommended_weight:.3f}")
            print(explanation)
            print("=" * 70 + "\n")

        # Online адаптація: ініціалізація EMA
        current_inv_weight = self.cfg.inv_weight
        inv_weight_history = [current_inv_weight]

        if self.cfg.adaptive_mode == "online" and self.adaptive_controller is not None:
            # Отримати початкову рекомендацію
            print("\n" + "=" * 70)
            print("🔍 ADAPTIVE INVERSION: Online mode - initial recommendation")
            print("=" * 70)
            recommended_weight, explanation = self.adaptive_controller.recommend(X_train, Y_train)
            current_inv_weight = recommended_weight
            inv_weight_history[0] = current_inv_weight
            print(f"Початковий inv_weight: {current_inv_weight:.3f}")
            print(explanation)
            print("=" * 70 + "\n")

            # Ініціалізація EMA
            base_loss_ema = None
            inv_loss_ema = None
            val_loss_ema = None
            prev_val_loss_ema = None

        # Тренувальний цикл
        for epoch in range(self.cfg.num_epochs):
            self.model.train()

            epoch_base_loss = 0.0
            epoch_inv_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0

            for batch_X, batch_Y in train_loader:
                self.optimizer.zero_grad()

                # Forward pass
                y_pred, h = self.model(batch_X, return_repr=self.cfg.use_representation_for_inv)

                # Базова втрата
                base_loss = self.criterion(y_pred, batch_Y)

                # Інверсійна втрата
                inv_loss = torch.tensor(0.0, device=self.device)
                if self.cfg.use_inversion and self.perturbation is not None:
                    # Створити збурений вхід
                    batch_X_perturbed = self.perturbation.apply(batch_X)

                    # Forward pass на збуреному вході
                    y_pred_perturbed, h_perturbed = self.model(
                        batch_X_perturbed, return_repr=self.cfg.use_representation_for_inv
                    )

                    # Обчислити інверсійну втрату
                    if self.cfg.use_representation_for_inv:
                        # Інверсія на репрезентаціях
                        inv_loss = self.criterion(h_perturbed, h.detach())
                    else:
                        # Інверсія на виходах
                        inv_loss = self.criterion(y_pred_perturbed, y_pred.detach())

                # Загальна втрата (використовуємо поточний динамічний inv_weight)
                total_loss = base_loss + current_inv_weight * inv_loss

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

                # Логування
                epoch_base_loss += base_loss.item()
                epoch_inv_loss += inv_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1

            # Усереднити втрати
            epoch_base_loss /= num_batches
            epoch_inv_loss /= num_batches
            epoch_total_loss /= num_batches

            # Валідація
            self.model.eval()
            with torch.no_grad():
                val_pred, _ = self.model(X_val_t, return_repr=False)
                val_loss = self.criterion(val_pred, Y_val_t).item()

            # Обчислити метрики
            gain = (epoch_base_loss - val_loss) / (epoch_base_loss + 1e-8)
            inv_eff = epoch_inv_loss / (epoch_base_loss + 1e-8) if self.cfg.use_inversion else 0.0

            # Online адаптація
            if self.cfg.adaptive_mode == "online" and self.adaptive_controller is not None:
                # Оновити EMA
                if base_loss_ema is None:
                    base_loss_ema = epoch_base_loss
                    inv_loss_ema = epoch_inv_loss
                    val_loss_ema = val_loss
                    prev_val_loss_ema = val_loss
                else:
                    prev_val_loss_ema = val_loss_ema
                    base_loss_ema = (
                        self.cfg.ema_alpha * base_loss_ema
                        + (1 - self.cfg.ema_alpha) * epoch_base_loss
                    )
                    inv_loss_ema = (
                        self.cfg.ema_alpha * inv_loss_ema
                        + (1 - self.cfg.ema_alpha) * epoch_inv_loss
                    )
                    val_loss_ema = (
                        self.cfg.ema_alpha * val_loss_ema + (1 - self.cfg.ema_alpha) * val_loss
                    )

                # Застосувати адаптацію після warmup кожні adaptive_step епох
                if (
                    epoch >= self.cfg.adaptive_warmup
                    and (epoch - self.cfg.adaptive_warmup) % self.cfg.adaptive_step == 0
                ):
                    new_weight, adaptation_reason = self.adaptive_controller.update_with_feedback(
                        epoch=epoch + 1,
                        base_loss_ema=base_loss_ema,
                        inv_loss_ema=inv_loss_ema,
                        val_loss_ema=val_loss_ema,
                        prev_val_loss_ema=prev_val_loss_ema,
                        current_inv_weight=current_inv_weight,
                        adaptive_eta=self.cfg.adaptive_eta,
                        adaptive_eta2=self.cfg.adaptive_eta2,
                        r_target=self.cfg.adaptive_r_target,
                    )

                    if abs(new_weight - current_inv_weight) > 0.01:
                        print(f"\n📊 Epoch {epoch+1}: Adaptive update")
                        print(f"   inv_weight: {current_inv_weight:.3f} → {new_weight:.3f}")
                        print(
                            f"   Reason: r={inv_loss_ema/(base_loss_ema+1e-12):.4f}, val_trend={val_loss_ema-prev_val_loss_ema:.6f}"
                        )

                    current_inv_weight = new_weight

                inv_weight_history.append(current_inv_weight)
            else:
                inv_weight_history.append(current_inv_weight)

            # Логування метрик
            epoch_metrics = {
                "epoch": epoch + 1,
                "base_loss": epoch_base_loss,
                "inv_loss": epoch_inv_loss,
                "total_loss": epoch_total_loss,
                "val_loss": val_loss,
                "gain": gain,
                "inv_eff": inv_eff,
                "inv_weight": float(current_inv_weight),
            }

            # Додати EMA метрики для online режиму
            if self.cfg.adaptive_mode == "online" and base_loss_ema is not None:
                epoch_metrics["base_loss_ema"] = float(base_loss_ema)
                epoch_metrics["inv_loss_ema"] = float(inv_loss_ema)
                epoch_metrics["val_loss_ema"] = float(val_loss_ema)
                epoch_metrics["ratio_r"] = float(inv_loss_ema / (base_loss_ema + 1e-12))

            # Зберегти в файл
            with open(metrics_file, "a") as f:
                f.write(json.dumps(epoch_metrics) + "\n")

            train_history.append(epoch_metrics)

            # Callback
            if callback is not None:
                callback(epoch, epoch_metrics)

            # Вивести прогрес
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{self.cfg.num_epochs}: "
                    f"base_loss={epoch_base_loss:.4f}, "
                    f"inv_loss={epoch_inv_loss:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )

            # Зберегти найкращу модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(run_dir, "best_model.pt"))

        # Генерувати візуалізацію inv_weight schedule
        self._plot_inv_weight_schedule(inv_weight_history, train_history, run_dir)

        # Зберегти фінальну інформацію про run
        run_info = {
            "config": asdict(self.cfg),
            "model_info": self.model.model_info(),
            "best_val_loss": best_val_loss,
            "final_train_loss": epoch_base_loss,
            "run_dir": run_dir,
            "tag": tag,
            "inv_weight_history": [float(w) for w in inv_weight_history],
        }

        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2)

        print(f"\nТренування завершено. Найкраща val_loss: {best_val_loss:.4f}")

        # Вивести пояснення адаптивного контролера
        if self.adaptive_controller is not None:
            explanation = self.adaptive_controller.explain()
            explanation_file = os.path.join(run_dir, "adaptive_explanation.txt")
            with open(explanation_file, "w", encoding="utf-8") as f:
                f.write(explanation)
            print(f"\nАдаптивне пояснення збережено в: {explanation_file}")

        return run_info

    def evaluate(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        noise_levels: list[float] = [0.0, 0.01, 0.05, 0.1],
    ) -> dict:
        """
        Оцінка робастності моделі до різних рівнів шуму.

        Args:
            X_test (np.ndarray): Тестові дані
            Y_test (np.ndarray): Тестові таргети
            noise_levels (List[float]): Рівні шуму для тестування

        Returns:
            Dict: Словник з результатами оцінки
        """
        self.model.eval()
        results = {}

        X_test_t = torch.FloatTensor(X_test).to(self.device)
        Y_test_t = torch.FloatTensor(Y_test).unsqueeze(1).to(self.device)

        with torch.no_grad():
            for noise_level in noise_levels:
                # Додати шум
                if noise_level > 0:
                    noise = torch.randn_like(X_test_t) * noise_level
                    X_noisy = X_test_t + noise
                else:
                    X_noisy = X_test_t

                # Передбачення
                y_pred, _ = self.model(X_noisy, return_repr=False)

                # Обчислити MSE
                mse = self.criterion(y_pred, Y_test_t).item()

                results[f"noise_{noise_level}"] = {"mse": mse, "rmse": np.sqrt(mse)}

        return results

    def _plot_inv_weight_schedule(
        self, inv_weight_history: list[float], train_history: list[dict], run_dir: str
    ):
        """
        Створює візуалізацію зміни inv_weight по епохах.

        Args:
            inv_weight_history (List[float]): Історія inv_weight
            train_history (List[Dict]): Історія тренування
            run_dir (str): Директорія для збереження
        """
        try:
            import matplotlib.pyplot as plt

            epochs = [m["epoch"] for m in train_history]
            val_losses = [m["val_loss"] for m in train_history]
            inv_losses = [m["inv_loss"] for m in train_history]

            # Переконатися що inv_weight_history має правильний розмір
            # Він може мати на 1 елемент більше через початкову ініціалізацію
            if len(inv_weight_history) > len(epochs):
                inv_weight_history = inv_weight_history[1:]  # Видалити перший елемент
            elif len(inv_weight_history) < len(epochs):
                # Якщо менше, додати останнє значення
                inv_weight_history = inv_weight_history + [inv_weight_history[-1]] * (
                    len(epochs) - len(inv_weight_history)
                )

            # Створити figure з двома subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Plot 1: inv_weight
            ax1.plot(epochs, inv_weight_history, "b-", linewidth=2, label="inv_weight")
            ax1.axhline(
                y=self.cfg.inv_weight_max,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"max={self.cfg.inv_weight_max}",
            )
            ax1.axhline(
                y=self.cfg.inv_weight_min,
                color="g",
                linestyle="--",
                alpha=0.5,
                label=f"min={self.cfg.inv_weight_min}",
            )
            if self.cfg.adaptive_warmup > 0:
                ax1.axvline(
                    x=self.cfg.adaptive_warmup,
                    color="orange",
                    linestyle=":",
                    alpha=0.7,
                    label=f"warmup={self.cfg.adaptive_warmup}",
                )
            ax1.set_ylabel("inv_weight", fontsize=12)
            ax1.set_title(
                f"Adaptive Inversion Schedule ({self.cfg.adaptive_mode} mode)",
                fontsize=14,
                fontweight="bold",
            )
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)

            # Plot 2: Losses
            ax2_twin = ax2.twinx()
            ax2.plot(epochs, val_losses, "g-", linewidth=2, label="val_loss", alpha=0.8)
            ax2_twin.plot(epochs, inv_losses, "r-", linewidth=2, label="inv_loss", alpha=0.8)

            ax2.set_xlabel("Epoch", fontsize=12)
            ax2.set_ylabel("Validation Loss", fontsize=12, color="g")
            ax2_twin.set_ylabel("Inversion Loss", fontsize=12, color="r")
            ax2.tick_params(axis="y", labelcolor="g")
            ax2_twin.tick_params(axis="y", labelcolor="r")
            ax2.grid(True, alpha=0.3)

            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            plt.tight_layout()

            save_path = os.path.join(run_dir, "inv_weight_schedule.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"📊 Візуалізація inv_weight збережена: {save_path}")

        except Exception as e:
            print(f"⚠️ Помилка при створенні візуалізації: {e}")

    def _generate_run_dir(self, tag: str) -> str:
        """
        Створює директорію для результатів run.

        Args:
            tag (str): Тег для run

        Returns:
            str: Шлях до директорії
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.cfg.data_mode}_{tag}_{timestamp}"
        run_dir = os.path.join(self.cfg.results_root, run_name)
        return run_dir
