# 📱 Options Premium Tracker

A professional-grade mobile app for tracking options straddle premiums in real-time. Built with Streamlit and the Dhan API.

## 🚀 Features

- Real-time options premium tracking
- Straddle premium visualization
- Mobile-first design
- EMA indicators (9 & 21)
- Previous day close reference
- Support for all F&O stocks

## 📋 Requirements

- Python 3.9+
- Dhan API credentials
- Required packages in `requirements_streamlit.txt`

## 🛠️ Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd options-premium-tracker
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements_streamlit.txt
```

4. Set up your Dhan API credentials:
   - Create `.streamlit/secrets.toml`
   - Add your credentials:
   ```toml
   DHAN_API = "your_api_key"
   DHAN_CLIENTID = "your_client_id"
   ```

5. Run the app:
```bash
streamlit run mobile_options_app.py
```

## 📊 Usage

1. Select a stock from the F&O list
2. Choose a strike price
3. Select timeframe (1/3/5/15 min)
4. Generate live chart

## 🔒 Security

- Never commit your API credentials
- Use environment variables or Streamlit secrets
- Keep your API keys secure

## 📝 License

MIT License