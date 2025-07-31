import pandas as pd
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    model_name="deepseek-r1-distill-llama-70b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Load your data
df = pd.read_csv("dirty_data.csv")

def detect_issues(_=None) -> str:
    """Analyze dataset and return issue report"""
    issues = []
    
    # 1. Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values:\n{missing[missing > 0].to_string()}")
    
    # 2. Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append(f"Duplicate rows: {dupes}")
    
    # 3. Data type issues
    type_issues = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Try parsing with specific date formats
                parsed = pd.to_datetime(df[col], format='mixed', errors='raise')
                type_issues.append(f"'{col}' appears to contain dates but isn't datetime")
            except:
                pass
    if type_issues:
        issues.append("Data type warnings:\n" + "\n".join(type_issues))
    
    return "\n\n".join(issues) if issues else "No major issues detected"

def clean_data(instruction: str) -> str:
    global df
    try:
        result = []
        
        # Handle "clean all" command
        if "clean all" in instruction.lower():
            # Remove duplicates
            before = df.shape[0]
            df.drop_duplicates(inplace=True)
            after = df.shape[0]
            result.append(f"Removed {before-after} duplicates")
            
            # Fill missing values
            if df.isnull().sum().sum() > 0:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col].fillna('Unknown', inplace=True)
                    elif df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                result.append("Filled all missing values")
            
            # Standardize dates
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], format='mixed').dt.strftime('%Y-%m-%d')
                        result.append(f"Standardized dates in {col}")
                    except:
                        continue
                        
            return ". ".join(result) + f". New shape: {df.shape}"
        
        # Handle specific commands
        if "remove duplicates" in instruction.lower():
            before = df.shape[0]
            df.drop_duplicates(inplace=True)
            after = df.shape[0]
            return f"Removed {before-after} duplicates. New shape: {df.shape}"
            
        return "No valid cleaning operation specified"
    except Exception as e:
        return f"Error: {str(e)}"

# Define tools
tools = [
    Tool(
        name="DataDiagnostic",
        func=detect_issues,
        description="Detects data quality issues"
    ),
    Tool(
        name="DataCleaning",
        func=clean_data,
        description="Performs cleaning operations. Use 'clean all' for comprehensive cleaning."
    ),
    Tool(
        name="DataPreview",
        func=lambda _: df.head().to_string(),
        description="Shows first 5 rows"
    )
]

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data cleaning assistant. Respond concisely and execute commands directly.
Available commands:
- analyze: Show data issues
- clean [operation]: Perform cleaning
- preview: Show data sample
- save: Export cleaned data"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Interactive session
print("🛠️ Data Cleaning Agent")
print("Commands: 'analyze', 'clean [operation]', 'preview', 'save', 'exit'")

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ('exit', 'quit'):
        break
        
    if user_input.lower() == 'save':
        df.to_csv("cleaned_data.csv", index=False)
        print("✅ Saved cleaned data to cleaned_data.csv")
        continue
        
    if user_input.lower() == 'preview':
        print(df.head())
        continue
        
    try:
        result = agent_executor.invoke({"input": user_input})
        # Clean up the output
        output = result['output'].replace('<tool_call>', '').replace('</tool_call>', '')
        print(f"\n🤖 Agent: {output}")
    except Exception as e:
        print(f"\n⚠️ Error: {str(e)}")