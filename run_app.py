import subprocess
import os
import sys
import signal

# Get the Streamlit port and host from environment
port = os.getenv("CDSW_APP_PORT", "8080")
host = os.getenv("CDSW_APP_HOST", "127.0.0.1")

# Launch Streamlit
process = subprocess.Popen([
    "streamlit",
    "run",
    "5_fraud_detection_app.py",
    "--server.port", port,
    "--server.address", host,
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
])

# Handle shutdown gracefully
def signal_handler(sig, frame):
    print("Shutting down Streamlit...")
    process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Wait for the process to complete
process.wait()