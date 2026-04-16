import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import time

os.environ["OPENAI_API_BASE"] = 'https://YOUR_LLM_API/v1'
os.environ["OPENAI_MODEL_NAME"] = 'qwen2b'  # Adjust based on available model
os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'

os.environ["SERPER_API_KEY"] = 'YOUR_SERPER_API_KEY'
search_tool = SerperDevTool()
from crewai import Agent
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper
from crewai_tools import WebsiteSearchTool
from program.tools.calculator_tools import CalculatorTools
from program.tools.browser_tools import BrowserTools

search = GoogleSerperAPIWrapper()
websearch_tool = WebsiteSearchTool()
Browsersearch_tool = BrowserTools.scrape_and_summarize_website,
calculate_tool = CalculatorTools.calculate
serper_tool = Tool(
  name="Online search tool",
  func=search.run,
  description="Useful for search-based queries",
)

class Agents():

  def KnowledgeQuery(self):
      return Agent(
          role='KnowledgeQuery',
          goal='Get information related to the query',
          backstory="""You are responsible for identifying all information and content related to the query.""",
          verbose=True,
          allow_delegation=False,
          tools=[serper_tool, search_tool, websearch_tool]
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
          tools=[calculate_tool]
        )

  def Code_generate(self):
      return Agent(
          role='Code(pseudo_code)',
          goal='Generate a Python function that corresponds to the pseudo code',
          backstory="""You have the ability to generate code and can generate Python code by referencing pseudocode. Of course, your ultimate goal is to solve user tasks.""",
          verbose=True,
          allow_delegation=False,
        )
