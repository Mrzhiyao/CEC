from crewai import Agent, Task, Crew, Process
from program.agents import Agents

def get_tasks(task_list):
    Tasks = []
    agents = Agents()
    # agents.KnowledgeQuery()
    for i in range(len(task_list)):
        task_content = task_list[i]
        sign = "R" + str(i+1)
        task_content = task_content.replace(sign + " = ", "")
        # print(task_content)
        if 'KnowledgeQuery' in task_content:
            task_content = task_content.replace("KnowledgeQuery", "")
            des = """Get information related to  """ + task_content
            Tasks.append(Task(
                description=des,
                expected_output="All relevant information related to the query",
                agent=agents.KnowledgeQuery()
            ))
        elif 'ParagraphRetrieve' in task_content:
            task_content = task_content.replace("ParagraphRetrieve", "")
            position = task_content.find('Query: ')
            task_content = task_content[position:-1]
            # print(task_content)
            des = """Extract information related to the query from the paragraph:\n """ + task_content
            Tasks.append(Task(
                description=des,
                expected_output="The information related to the query",
                agent=agents.ParagraphRetrieve()
            ))
        elif 'QA' in task_content:
            task_content = task_content.replace("QA", "")
            position = task_content.find('Question: ')
            task_content = task_content[position:-1]
            # print(task_content)
            des = """Accurately answer questions based on extracted knowledge: """ + task_content
            Tasks.append(Task(
                description=des,
                expected_output="Accurate answers related to the query",
                agent=agents.QA()
            ))
        elif 'Calculator' in task_content:
            task_content = task_content.replace("Calculator", "")
            des = """Calculate the input formula """ + task_content
            Tasks.append(Task(
                description=des,
                expected_output="Get the calculation results described in the query",
                agent=agents.Calculater()
            ))
        elif 'Code' in task_content:
            task_content = task_content.replace("Code", "")
            position = task_content.find('Query: ')
            task_content = task_content[position:-2]
            # print(task_content)
            des = """Generate a Python function that corresponds to the pseudo code """ + task_content
            Tasks.append(Task(
                description=des,
                expected_output="Python program corresponding to pseudocode",
                agent=agents.Code_generate()
            ))

    return Tasks
    # task1 = Task(
    #     description="""Obtain all relevant information related to the "Hello Love".""",
    #     expected_output="All relevant information related to the problem",
    #     agent=agents.KnowledgeQuery()
    # )
    #
    # task2 = Task(
    #     description="""Find the answer to this question:\n Who is the performer of "Hello Love"?.""",
    #     expected_output="Accurate answers related to the question",
    #     agent=agents.ParagraphRetrieve()
    # )
