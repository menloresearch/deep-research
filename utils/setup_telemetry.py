#!/usr/bin/env python3
"""
Helper script to setup telemetry for the research agent.
Choose between Phoenix (local) or Langfuse (cloud) backends.
"""

import os
import subprocess
import sys
import base64

def install_phoenix():
    """Install and setup Phoenix for local telemetry"""
    print("Installing Phoenix dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents[telemetry]", "arize-phoenix"])
        print("✅ Phoenix installed successfully!")
        print("\nTo start Phoenix server, run:")
        print("python -m phoenix.server.main serve")
        print("\nThen view traces at: http://0.0.0.0:6006/projects/")
        print("\nAfter starting Phoenix, run your research agent normally.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Phoenix: {e}")
        return False

def setup_langfuse():
    """Setup Langfuse for cloud telemetry"""
    print("Installing Langfuse dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "smolagents[telemetry]", 
            "opentelemetry-sdk", 
            "opentelemetry-exporter-otlp"
        ])
        print("✅ Langfuse dependencies installed!")
        
        print("\nLangfuse setup requires API keys:")
        print("1. Sign up at https://cloud.langfuse.com/")
        print("2. Get your API keys from the dashboard")
        
        public_key = input("Enter your Langfuse public key (pk-lf-...): ").strip()
        secret_key = input("Enter your Langfuse secret key (sk-lf-...): ").strip()
        
        if not public_key.startswith("pk-lf-") or not secret_key.startswith("sk-lf-"):
            print("❌ Invalid API keys format")
            return False
            
        # Create base64 auth
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        
        # Set environment variables
        env_content = f"""
# Langfuse telemetry configuration
export OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic {auth}"
"""
        
        # Write to .env file
        with open(".env.langfuse", "w") as f:
            f.write(env_content.strip())
            
        print("✅ Langfuse configuration saved to .env.langfuse")
        print("\nTo use Langfuse, source the environment file:")
        print("source .env.langfuse")
        print("Then run your research agent normally.")
        print("\nView traces at: https://cloud.langfuse.com/")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Langfuse dependencies: {e}")
        return False
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled")
        return False

def main():
    print("🔍 Research Agent Telemetry Setup")
    print("=" * 40)
    print("Choose telemetry backend:")
    print("1. Phoenix (Local, simple setup)")
    print("2. Langfuse (Cloud, advanced features)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            success = install_phoenix()
            break
        elif choice == "2":
            success = setup_langfuse()
            break
        elif choice == "3":
            print("Setup cancelled")
            return
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("Your research agent will now provide comprehensive tracing.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main() 