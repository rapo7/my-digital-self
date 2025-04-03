# RaviGPT - Streamlit Version

A Streamlit-based personal AI assistant that answers questions about Ravi, a Software Engineer. This is a Python conversion of the original React-based application.

## Features

- **Interactive Chat Interface**: Ask questions about Ravi's background, experience, projects, and interests
- **Category Selection**: Browse different categories of information with an intuitive button interface
- **Suggestion System**: Get question suggestions based on the selected category
- **Markdown Formatting**: Responses are formatted with rich markdown for readability
- **Loading Animation**: Visual feedback while waiting for responses

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will launch in your default web browser at http://localhost:8501.

## Customization

You can easily customize the responses by modifying the `get_ravi_response` function in `app.py`. In a production version, this would connect to an AI service or database with information about Ravi.

## Key Differences from the React Version

- **Simplified Stack**: Uses Python instead of JavaScript/TypeScript
- **Built-in Components**: Leverages Streamlit's native chat, button, and markdown components
- **State Management**: Uses Streamlit's session state instead of React's useState
- **Styling**: Uses CSS injected via Streamlit's markdown capabilities

## Future Improvements

- Connect to an LLM API for more dynamic responses
- Add user authentication for personalized experiences
- Implement caching for faster responses
- Add a sidebar with additional information and settings
