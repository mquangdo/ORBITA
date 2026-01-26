from langchain_core.messages import HumanMessage
from manager_agent import manager_agent
from opik import configure 
from opik.integrations.langchain import OpikTracer


configure() 

def run():
    project_name = 'ORBITA'
    tracer = OpikTracer(graph=manager_agent.get_graph(xray=True), project_name=project_name) 
    while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            inputs = {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            }

            result = manager_agent.invoke(inputs, config={
                "callbacks": [tracer], 
            })
            print("Manager Agent:", result['messages'][-1].content)
            
if __name__ == "__main__":
    run()