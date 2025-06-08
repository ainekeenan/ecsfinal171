
Set up a Virtual Environment
- Open Command Prompt (Windows) or Terminal (macOS/Linux).
- Navigate to your project folder by running:
  - `cd path\to\your\project` (Windows)
  - `cd /path/to/your/project` (macOS/Linux)
- Create a virtual environment by running:
  - `python -m venv venv`
- Activate the virtual environment:
  - For Windows: `venv\Scripts\activate`
  - For macOS/Linux: `source venv/bin/activate`

Install Flask
- After activating the virtual environment, install Flask by running:
  - `pip install Flask`

Run the Application
- In your terminal, within the project directory, run:
  - `python app.py`
- Open your web browser and navigate to:
  - `http://localhost:5000`
- You should see the HTML form displayed.
