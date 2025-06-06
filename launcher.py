import subprocess
import sys
import os

def run_streamlit_app():
    try:
        print("🚀 Starting Company Intelligence Platform...")
        app_path = os.path.join(os.path.dirname(__file__), "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    run_streamlit_app()