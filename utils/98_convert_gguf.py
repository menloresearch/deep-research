#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB"""
    return Path(file_path).stat().st_size / (1024**3)

def convert_to_gguf(model_dir: str, output_dir: str, q_method: str = "Q4_K_M"):
    # Extract model name from the directory path
    model_name = Path(model_dir).name
    output_name = f"{model_name.lower()}-{q_method.lower()}.gguf"
    
    print(f"Converting {model_dir} to GGUF format with {q_method} quantization...")
    print(f"Output file: {output_name}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert to f16 GGUF first
    fp16_path = Path(output_dir) / f"temp-f16.gguf"
    final_path = Path(output_dir) / output_name
    
    print("Step 1: Converting to f16 GGUF...")
    try:
        cmd = [
            "python", "llama.cpp/convert_hf_to_gguf.py", 
            model_dir,
            "--outtype", "f16",
            "--outfile", str(fp16_path)
        ]
        subprocess.run(cmd, check=True)
        f16_size = get_file_size_gb(str(fp16_path))
        print(f"✓ Converted to f16: {fp16_path} ({f16_size:.1f}GB)")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ F16 conversion failed: {e}")
        sys.exit(1)
    
    # Check if llama-quantize exists (try both old and new locations)
    quantize_paths = [
        "./llama.cpp/build/bin/llama-quantize",  # CMake build location
        "./llama.cpp/llama-quantize",            # Old make location
        "./llama.cpp/build/llama-quantize"       # Alternative CMake location
    ]
    
    quantize_binary = None
    for path in quantize_paths:
        if Path(path).exists():
            quantize_binary = path
            break
    
    if not quantize_binary:
        print(f"❌ llama-quantize not found in any of these locations:")
        for path in quantize_paths:
            print(f"   {path}")
        print("💡 Run: make gguf-build")
        print("🔄 Falling back to f16 version...")
        fp16_path.rename(final_path)
        final_size = get_file_size_gb(str(final_path))
        print(f"⚠️  WARNING: Using f16 version ({final_size:.1f}GB) - should be ~8GB for Q4_K_M")
        return str(final_path)
    
    # Step 2: Quantize to desired method
    print(f"Step 2: Quantizing to {q_method}...")
    try:
        cmd = [
            quantize_binary,
            str(fp16_path),
            str(final_path),
            q_method
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if final_path.exists():
            final_size = get_file_size_gb(str(final_path))
            print(f"✓ Quantized to {final_path} ({final_size:.1f}GB)")
            
            # Validate size - Q4_K_M should be much smaller
            if final_size > 15:  # If still >15GB, something's wrong
                print(f"⚠️  WARNING: File size {final_size:.1f}GB seems too large for {q_method}")
                print(f"Expected ~8GB for 14B model with {q_method}")
            
            # Clean up temp f16 file
            if fp16_path.exists():
                fp16_path.unlink()
                print("✓ Cleaned up temporary f16 file")
            
            return str(final_path)
        else:
            raise Exception("Quantized file not created")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Quantization failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error details: {e.stderr}")
        print("🔄 Falling back to f16 version...")
        
        # If quantization fails, rename f16 to final name
        if fp16_path.exists():
            fp16_path.rename(final_path)
            final_size = get_file_size_gb(str(final_path))
            print(f"⚠️  WARNING: Using f16 version ({final_size:.1f}GB) - should be ~8GB for Q4_K_M")
            return str(final_path)
        sys.exit(1)

def main():
    # Default values
    default_model_dir = "temp_model"
    default_output_dir = "gguf_output"
    default_q_method = "Q4_K_M"
    
    if len(sys.argv) == 1:
        # No arguments, use defaults
        model_dir, output_dir, q_method = default_model_dir, default_output_dir, default_q_method
    elif len(sys.argv) == 3:
        # Model dir, output dir provided
        model_dir, output_dir = sys.argv[1:3]
        q_method = default_q_method
    elif len(sys.argv) == 4:
        # All arguments provided
        model_dir, output_dir, q_method = sys.argv[1:4]
    else:
        print("Usage: python 98_convert_gguf.py [model_dir] [output_dir] [q_method]")
        print(f"Defaults: {default_model_dir} {default_output_dir} {default_q_method}")
        print("Available quantization methods: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0")
        print("Output filename will be: <model_name>-<q_method>.gguf")
        sys.exit(1)
    
    result_path = convert_to_gguf(model_dir, output_dir, q_method)
    final_size = get_file_size_gb(result_path)
    print(f"🎉 Conversion complete: {result_path} ({final_size:.1f}GB)")

if __name__ == "__main__":
    main()