# 🧠 Multi-Model AI Chatbot (OpenRouter)

A powerful Streamlit chatbot that lets you compare responses from 95+ FREE and ultra-cheap AI models through OpenRouter's unified API.

## ✨ Features

- **95+ Models**: Access to FREE models and ultra-cheap options (starting at $0.02/1M tokens)
- **Model Browser**: Explore models by provider, capability, and cost
- **Parallel Processing**: Get responses from multiple models simultaneously  
- **Large Free Limits**: Up to 1000 requests/day for FREE models
- **Smart UI**: Tabs, columns, or stacked layouts for multiple models
- **Cost Tracking**: See exact costs and limits for each model
- **Export/Import**: Save conversations and transfer model selections

## 📁 Project Structure

```
├── app.py              # Main application entry point
├── models.py           # Model definitions and categorization
├── api_utils.py        # API calls and utility functions
├── browser.py          # Model browser interface
├── chat.py             # Chat interface
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create this)
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone <your-repo>
cd openrouter-free-models
pip install -r requirements.txt
```

### 2. Get API Key
1. Visit [OpenRouter](https://openrouter.ai/keys)
2. Create a free account
3. Generate an API key

### 3. Configure Environment
Create a `.env` file:
```bash
OPENROUTER_API_KEY=sk-or-your-api-key-here
```

### 4. Run the App
```bash
streamlit run app.py
```

## 💡 Understanding Costs & Limits

### FREE Models (:free)
- **Cost**: Completely FREE
- **Rate Limits**: 20 requests/minute
- **Daily Limits**: 
  - 50 requests/day (basic account)
  - 1000 requests/day (after buying $10+ credits)

### Ultra-Cheap Models
- **Cost**: $0.02 - $0.08 per million tokens
- **Example**: A 1000-word article = ~1000 tokens = $0.02-0.08
- **No rate limits**

### How to Get Large Free Limits
1. **Use FREE models** - Completely free with rate limits
2. **Buy $10 credits** - Increases free model limits from 50 to 1000 req/day
3. **Mix with ultra-cheap** - Use $0.02-0.08 models for high volume

## 🏆 Top Model Recommendations

| Category | Model | Cost | Description |
|----------|-------|------|-------------|
| 🚀 **Fastest** | Llama 3.2 1B | FREE | Smallest, fastest responses |
| 🧠 **Reasoning** | DeepSeek R1 | FREE | Advanced reasoning capabilities |
| 🏆 **Most Capable** | Llama 3.3 70B | FREE | Best overall performance |
| 💻 **Coding** | Optimus Alpha | FREE | 1M context coding specialist |
| 👁️ **Vision** | Llama 3.2 11B Vision | FREE | Image understanding |
| 🦣 **Largest** | Llama 4 Maverick | FREE | 400B MoE parameters |
| 💰 **Cheapest** | Ministral 3B | $0.04/1M | Ultra-affordable |

## 🔧 Usage Tips

### Model Browser
- **Search**: Find models by name or capability
- **Categories**: Browse by provider (DeepSeek, Meta, Google, etc.)
- **Capabilities**: Filter by reasoning, vision, coding, etc.
- **Transfer**: Select models and transfer to chat

### Chat Interface
- **Multiple Models**: Compare up to 10+ models side-by-side
- **Layouts**: Choose tabs, columns, or stacked view
- **System Messages**: Set AI personality globally
- **Parallel Processing**: All models respond simultaneously

### API Key Management
- **Environment**: Load from .env file (recommended)
- **Manual**: Enter directly in the app
- **Validation**: Real-time format checking
- **Testing**: Built-in connection test

## 🛠️ Development

### File Descriptions

- **`app.py`**: Main entry point, sets up tabs and navigation
- **`models.py`**: Contains all model definitions and categorization logic
- **`api_utils.py`**: API calling functions with error handling and parallel processing
- **`browser.py`**: Model browser interface with search and filtering
- **`chat.py`**: Chat interface with sidebar controls and conversation management

### Adding New Models

Edit `models.py` and add to the `MODEL_OPTIONS` dictionary:

```python
"provider/model-name:free": "Display Name (FREE)",
"provider/model-name": "Display Name ($0.XX/1M)",
```

### Customizing Categories

Modify the `categorize_models()` and `analyze_model_capabilities()` functions in `models.py`.

## 📊 Current Model Count

- **Total Models**: 95+
- **FREE Models**: 80+
- **Ultra-Cheap Models**: 8+
- **Providers**: 15+

## 🔗 Useful Links

- [OpenRouter Dashboard](https://openrouter.ai/)
- [API Documentation](https://openrouter.ai/docs)
- [Model Pricing](https://openrouter.ai/models)
- [Usage Tracking](https://openrouter.ai/usage)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## ⚠️ Important Notes

- **Rate Limits**: FREE models have daily limits
- **Credits**: $10 minimum purchase increases FREE limits dramatically
- **API Keys**: Keep your API keys secure and never commit them
- **Costs**: Ultra-cheap models have tiny costs but no rate limits
- **Updates**: Model availability can change on OpenRouter

---

Made with ❤️ using Streamlit and OpenRouter
