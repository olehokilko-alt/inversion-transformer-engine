import os
import sys
import shutil
import subprocess
import glob
import zipfile

def build_secure_dist():
    print("🚀 Starting Secure Distribution Build...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(base_dir, "dist")
    
    # 1. Run Cython Compilation
    # Запускаємо компіляцію 'inplace', щоб згенерувати .pyd/.so файли поруч з .py
    print("\n🔨 Compiling Core Modules...")
    try:
        subprocess.check_call([sys.executable, "setup_cython.py", "build_ext", "--inplace"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Compilation failed: {e}")
        sys.exit(1)
        
    # 2. Prepare Dist Directory
    print(f"\n📂 Creating dist directory: {dist_dir}")
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    os.makedirs(dist_dir)
    
    # 3. Copy Directories (Assets, Docs, Serve)
    dirs_to_copy = ["assets", "docs", "serve"]
    for d in dirs_to_copy:
        src = os.path.join(base_dir, d)
        dst = os.path.join(dist_dir, d)
        if os.path.exists(src):
            print(f"   -> Copying {d}...")
            shutil.copytree(src, dst)
            
    # 4. Copy Files (Requirements, License, etc.)
    files_to_copy = ["requirements.txt", "README.md", "LICENSE"]
    for f in files_to_copy:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            print(f"   -> Copying {f}...")
            shutil.copy2(src, dist_dir)
            
    # 5. Copy Dockerfile.dist as Dockerfile
    # Для клієнта це буде основний Dockerfile
    docker_dist_src = os.path.join(base_dir, "Dockerfile.dist")
    if os.path.exists(docker_dist_src):
        print("   -> Copying Dockerfile.dist as Dockerfile...")
        shutil.copy2(docker_dist_src, os.path.join(dist_dir, "Dockerfile"))
    
    # 6. Handle Core (Secure Copy)
    print("\n🔒 Securing Core Logic...")
    core_src = os.path.join(base_dir, "core")
    core_dst = os.path.join(dist_dir, "core")
    os.makedirs(core_dst)
    
    # Копіюємо __init__.py
    shutil.copy2(os.path.join(core_src, "__init__.py"), core_dst)
    
    # Копіюємо perturbations.py (він залишається відкритим, як вказано в setup_cython.py)
    shutil.copy2(os.path.join(core_src, "perturbations.py"), core_dst)
    
    # Копіюємо ТІЛЬКИ скомпільовані розширення (.pyd для Windows, .so для Linux)
    compiled_extensions = []
    compiled_extensions.extend(glob.glob(os.path.join(core_src, "*.pyd")))
    compiled_extensions.extend(glob.glob(os.path.join(core_src, "*.so")))
    
    if not compiled_extensions:
        print("⚠️  WARNING: No compiled extensions found! Build might be broken.")
    
    for ext in compiled_extensions:
        filename = os.path.basename(ext)
        print(f"   -> Copying compiled module: {filename}")
        shutil.copy2(ext, core_dst)
        
    # 7. Verification (Critical Step)
    print("\n🕵️  Verifying Security...")
    forbidden_files = ["adaptive_controller.py", "inversion_transformer.py"]
    secure = True
    for f in forbidden_files:
        if os.path.exists(os.path.join(core_dst, f)):
            print(f"❌ SECURITY FAILURE: Source file {f} found in dist/core!")
            secure = False
        else:
            print(f"   ✅ Verified absence of {f}")
            
    if not secure:
        print("❌ Build Aborted due to security check failure.")
        sys.exit(1)
        
    # 8. Zip
    zip_filename = "Inversion_Transformer_Enterprise_v1.0_DIST.zip"
    print(f"\n📦 Zipping distribution to {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dist_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dist_dir)
                zipf.write(file_path, arcname)
                
    print(f"\n✅ Build Complete! Distribution archive: {zip_filename}")

if __name__ == "__main__":
    build_secure_dist()
