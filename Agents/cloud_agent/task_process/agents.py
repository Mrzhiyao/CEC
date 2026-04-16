import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai_tools import tool
import time

os.environ["OPENAI_API_BASE"] = 'http://YOUR_CLOUD_LLM_IP:9991/v1'
os.environ["OPENAI_MODEL_NAME"] = 'qwen14b'  # Adjust based on available model
os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'
os.environ["OTEL_SDK_DISABLED"] = "true"

os.environ["SERPER_API_KEY"] = 'YOUR_SERPER_API_KEY'
search_tool = SerperDevTool()
from crewai import Agent
from langchain.agents import Tool
from crewai_tools import WebsiteSearchTool
from langchain.tools import DuckDuckGoSearchRun

websearch_tool = WebsiteSearchTool()
ducksearch_tool = Tool(
    name="Web Search",
    func=DuckDuckGoSearchRun().run,
    description="useful for when you need to answer questions about current events"
)

@tool('DuckDuckGoSearch')
def search(search_query: str):
    """useful for when you need to answer questions about current events"""
    return DuckDuckGoSearchRun().run(search_query)

class Agents():

  def KnowledgeQuery(self):
      return Agent(
          role='KnowledgeQuery',
          goal='Get information related to the query',
          backstory="""You are responsible for identifying all information and content related to the query.""",
          verbose=True,
          allow_delegation=False,
          tools=[ducksearch_tool]
        )

  def ParagraphRetrieve(self):
      return Agent(
          role='ParagraphRetrieve',
          goal='Given a query, retrieve the most relevant paragraphs from the given context',
          backstory="""You need to accomplish the following tasks:\n Given a query, retrieve the most relevant paragraphs
          from the given context.Please read the reference information carefully and get accurate answers to your questions. If you need to search online, please get the information from the website in the relevant materials.""",
          verbose=True,
          allow_delegation=False,
        )
  def QA(self):
      return Agent(
          role='QA agent',
          goal='You need to find information related to the question from relevant information and provide answers',
          backstory="""You have the ability to answer questions and summarize information, and can accurately complete tasks based on the relevant information provided to you.""",
          verbose=True,
          allow_delegation=False,
        )

  def Calculater(self):
      return Agent(
          role='Calculator(formula)',
          goal='Calculate the input formula',
          backstory="""You have calculation function and complete calculation tasks based on prompt information.""",
          verbose=True,
          allow_delegation=False,
        )

  def Code_generate(self):
      return Agent(
          role='Code(pseudo_code)',
          goal='Generate a Python function that corresponds to the pseudo code',
          backstory="""You have the ability to generate code and can generate Python code by referencing pseudocode. Of course, your ultimate goal is to solve user tasks.""",
          verbose=True,
          allow_delegation=False,
        )

  def Comprehensive(self):
      return Agent(
          role='An agent that can solve user problems through Action Web Search',
          goal='If you can answer user questions, do so directly, otherwise seek solutions through online searches etc.',
          backstory="""You are a powerful agent designed to solve user tasks. You can only solve user tasks or reference knowledge to solve user queries through Action Web Search.""",
          verbose=True,
          allow_delegation=False,
          tools=[ducksearch_tool]
        )

  def Comprehensive_without(self):
      return Agent(
          role='An agent that can solve user problems through reference information',
          goal='If the reference information is relevant to the problem, please solve the problem based on the reference information. If the reference information is not related to the problem, please solve the problem directly.',
          backstory="""You are a powerful agent designed to solve user tasks. You can only solve user tasks or reference knowledge to solve user queries.""",
          verbose=True,
          allow_delegation=False,
        )
