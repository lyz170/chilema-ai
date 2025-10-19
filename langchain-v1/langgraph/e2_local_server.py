## 1. Install the LangGraph CLI - Python >= 3.11 is required.
# pip install -U "langgraph-cli[inmem]"

## 2. Create a LangGraph app
## Create a new app from the new-langgraph-project-python template.
## This template demonstrates a single-node application you can extend with your own logic.
# mkdir local-server
# cd local-server
# langgraph new . --template new-langgraph-project-python

## 3. Install dependencies
# pip install -e .

## 4. Create a .env file

## 5. Launch LangGraph Server
# langgraph dev
