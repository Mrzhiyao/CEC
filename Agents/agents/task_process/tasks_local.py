from crewai import Agent, Task, Crew, Process
from agents_local import Agents_local
import os 
os.environ["OTEL_SDK_DISABLED"] = "true"
def get_tasks_local(task):
    Tasks = []
    agents = Agents_local()
    task_content = task
    number_R = 0
    for i in range(10):
        sign = "R" + str(i)
        if sign in task:
            number_R = number_R + 1
    if number_R>=2:
        des = """Refer to the task process and complete the answer task:\n""" + task_content
        Tasks.append(Task(
            description=des,
            expected_output="Task answers that take into account relevant knowledge in the task process",
            agent=agents.Comprehensive()
        ))
    else:
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
        else:
            des = """Accurately answer questions based on extracted knowledge: """ + task_content
            Tasks.append(Task(
                description=des,
                expected_output="Accurate answers related to the query",
                agent=agents.Comprehensive()
            ))


    return Tasks

def get_tasks_conclusion_local(task):
    Tasks = []
    agents = Agents_local()
    task_content = task
    if 'QA' in task_content:
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
