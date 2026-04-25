import subprocess
import os
import sys   # ⭐ IMPORTANT: lets us use the current venv python

# Absolute path to Seed-VC repo
SEEDVC_DIR = r"D:\SVC\seed-vc"

def run_seedvc(vocal_path, target_path, workspace, source_gender="Male", target_gender="Male"):

    output_dir = os.path.join(workspace, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Convert all paths to absolute (SeedVC prefers this)
    vocal_path = os.path.abspath(vocal_path)
    target_path = os.path.abspath(target_path)
    output_dir = os.path.abspath(output_dir)

    # ⭐ Use the SAME Python interpreter that is running Streamlit
    python_exe = sys.executable

    print("Launching Seed-VC from:", SEEDVC_DIR)
    print("Using Python interpreter:", python_exe)
    
    # Configure parameters based on source/target gender mapping
    if source_gender == "Male" and target_gender == "Male":
        diffusion_steps = "40"
        semi_tone_shift = "0"
    elif source_gender == "Male" and target_gender == "Female":
        diffusion_steps = "60"
        semi_tone_shift = "12"
    elif source_gender == "Female" and target_gender == "Female":
        diffusion_steps = "40"
        semi_tone_shift = "0"
    elif source_gender == "Female" and target_gender == "Male":
        diffusion_steps = "60"
        semi_tone_shift = "-12"
    else:
        diffusion_steps = "40"
        semi_tone_shift = "0"

    cmd = [
        python_exe, "inference.py",   # ⭐ replaces "python"
        "--source", vocal_path,
        "--target", target_path,
        "--output", output_dir,
        "--f0-condition", "True",
        "--semi-tone-shift", semi_tone_shift,
        "--length-adjust", "1.0",
        "--diffusion-steps", diffusion_steps,
        "--inference-cfg-rate", "0.7",
        "--fp16", "True"
    ]

    # Run inside SeedVC directory so relative imports work
    subprocess.run(
        cmd,
        cwd=SEEDVC_DIR,
        check=True
    )

    return output_dir
