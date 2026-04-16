import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import time

os.environ["OPENAI_API_BASE"] = 'http://YOUR_LOCAL_LLM_IP:12555/v1'
os.environ["OPENAI_MODEL_NAME"] = 'qwen:1.8b'  # Adjust based on available model
os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'
os.environ["OTEL_SDK_DISABLED"] = "true"

os.environ["SERPER_API_KEY"] = 'YOUR_SERPER_API_KEY'
search_tool = SerperDevTool()
from crewai import Agent
from langchain.agents import Tool
from crewai_tools import WebsiteSearchTool
from tools.calculator_tools import CalculatorTools
from tools.browser_tools import BrowserTools

websearch_tool = WebsiteSearchTool()
Browsersearch_tool = BrowserTools.scrape_and_summarize_website,
calculate_tool = CalculatorTools.calculate

class Agents_local():

  def KnowledgeQuery(self):
      return Agent(
          role='KnowledgeQuery',
          goal='Get information related to the query',
          backstory="""You are responsible for identifying all information and content related to the query.""",
          verbose=True,
          allow_delegation=False,
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
          role='An agent that can solve user problems',
          goal='If you can answer user questions, do so directly, otherwise seek solutions through yourself',
          backstory="""You are a powerful agent designed to solve user tasks. You can solve user tasks or supplement knowledge through search to solve user problems.""",
          verbose=True,
          allow_delegation=False,
        )
