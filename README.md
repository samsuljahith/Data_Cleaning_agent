# Data Cleaning Agent — LangChain + Groq

An interactive CLI agent that diagnoses and cleans messy datasets using natural language commands. Built with LangChain tool-calling and Groq (DeepSeek R1), it wraps pandas operations as agent tools so you can clean data by describing what you want rather than writing code.

## Tools

| Tool | What It Does |
|------|-------------|
| `DataDiagnostic` | Scans for missing values, duplicates, and type issues |
| `DataCleaning` | Executes cleaning operations (`clean all`, `remove duplicates`, date standardisation) |
| `DataPreview` | Shows the first 5 rows of the current dataframe |

## Usage

```bash
pip install langchain langchain-groq pandas python-dotenv

# .env
GROQ_API_KEY=your_key
```

```bash
python data_cleaner.py
```

### Example Commands

```
analyze          → show all data quality issues
clean all        → remove duplicates, fill nulls, standardise dates
remove duplicates
preview
save             → export to cleaned_data.csv
exit
```

## Tech Stack

- **LLM** — Groq `deepseek-r1-distill-llama-70b`
- **Agent** — LangChain `create_tool_calling_agent` + `AgentExecutor`
- **Data** — pandas

## Input / Output

- Input: `dirty_data.csv` (place in project root)
- Output: `cleaned_data.csv` (after `save` command)
