from dotenv import load_dotenv
import os
load_dotenv()

from app import create_app
app = create_app()

if __name__ == "__main__":
    print("hi")
    app.run(host="0.0.0.0", port=5000)
