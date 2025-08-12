

# Data Analysis and Visualization AI Agent 📈

This project is an AI-powered agent designed to interact with a local dataset. It can understand natural language prompts to query data, perform analysis, and generate visualizations. The agent uses OpenAI's function-calling capabilities to intelligently route requests to a suite of specialized tools.

The primary example uses the **Baltimore City Employee Salaries (FY2013)** dataset to demonstrate how the agent can help organizations understand trends and anomalies in their data.

## ✨ Features

  * **🤖 Smart Tool Routing:** Uses an OpenAI model as a router to intelligently select the appropriate tool (query, analyze, or visualize) based on the user's prompt.
  * **🔍 Natural Language Database Querying:** Translates plain English questions into precise SQL queries to fetch data from a local CSV file using DuckDB.
  * **📊 AI-Powered Data Analysis:** Leverages a language model to analyze the queried data and provide textual insights and summaries.
  * **🎨 Automated Chart Generation:** Generates and executes Python code to create visualizations (e.g., bar charts) of the data using `matplotlib` and `seaborn`.
  * **🔬 Observability:** Integrated with **Arize Phoenix** to trace and monitor the agent's execution flow, including LLM calls and tool usage.

-----

## ⚙️ How It Works

The agent operates through a simple but powerful loop. When a user provides a prompt, the agent assesses it and decides which tool to use. After a tool runs, its output is fed back into the agent, which then decides the next step—either calling another tool or generating a final answer.

The agent is composed of three main tools:

1.  **Database Lookup (`lookup_salary_data`)**:

      * Takes a user prompt (e.g., "who are the highest-paid employees?").
      * Uses an LLM to convert the prompt into an SQL query.
      * Executes the query on the dataset loaded into an in-memory DuckDB table.
      * Returns the data as a string.

2.  **Data Analysis (`analyze_salary_data`)**:

      * Receives data from the lookup tool.
      * Uses an LLM to analyze the data and generate textual insights based on the original prompt.
      * Returns the analysis as a string.

3.  **Data Visualization (`generate_visualization`)**:

      * Receives data and a visualization goal (e.g., "plot this as a bar chart").
      * Uses an LLM to determine the best chart configuration (chart type, axes, title).
      * Uses another LLM call to generate Python code for the chart.
      * Executes the code to display the visualization.

-----

## 🚀 Getting Started

Follow these steps to set up and run the agent on your local machine.

### 1\. Prerequisites

  * Python 3.8+
  * An OpenAI API Key
  * An Arize Phoenix API Key and Collector Endpoint (for tracing)

### 2\. Installation

First, clone the repository and install the required Python packages.

```bash
# Clone the repository (if applicable)
# git clone https://your-repo-url.git
# cd your-repo-directory

# Install dependencies
pip install openai pandas duckdb pydantic phoenix openinference-instrumentation-openai opentelemetry-sdk seaborn matplotlib
```

### 3\. Download the Dataset

Download the Baltimore City Employee Salaries dataset and place it in a known directory.

  * **Download Link:** [Baltimore City Employee Salaries FY2013 on Kaggle](https://www.google.com/search?q=https://www.kaggle.com/datasets/datasciencedon/data-science-salaries)

### 4\. Configuration

Open the Python script and update the following configuration variables with your credentials and file path:

```python
# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Set your Phoenix tracing credentials (optional, but recommended)
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "your_phoenix_api_key"
os.environ["PHOENIX_CLIENT_HEADERS"] = "your_phoenix_api_key"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# Set the path to your downloaded dataset
SALARY_DATA_FILE_PATH = '/path/to/your/Baltimore_City_Employee_Salaries_FY2013.csv'

# Define the model you want to use
MODEL = "gpt-4o-mini"
```

-----

## 💻 Usage

To run the agent, simply execute the script. The main logic is initiated at the bottom of the file. You can modify the prompt inside the `start_main_span` function call to ask the agent different questions.

```python
# To run the script from your terminal
python your_script_name.py
```

### Example Prompt

You can change the `content` of the user message to ask your own questions. The example included in the script is:

```python
# This is the final call in the script that runs the agent
result = start_main_span([
    {
        "role": "user",
        "content": "Determine which job titles have the highest average Gross value in the given dataset and provide top 10 ranks overview and plot them as horizontal bar graph using seaborn."
    }
])

# The script will print the final analysis and display the generated chart
print(result)
```

The agent will process this request by:

1.  Calling `lookup_salary_data` to query the top 10 job titles by average gross pay.
2.  Calling `generate_visualization` to create a horizontal bar chart of the result.
3.  Calling `analyze_salary_data` to provide a textual summary.
4.  Returning the final compiled response.
