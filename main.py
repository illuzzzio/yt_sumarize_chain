from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import YouTubeSearchTool

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,  # Best practice for Gemini 3.0/2.5+ models to prevent reasoning degradation
    max_tokens=None,
    timeout=None,
    max_retries=2
)

class GraphNode(BaseModel):
    video_url : str = Field(description="The url for the yt videos ")
    video_id : Optional[str] = Field(default=None, description="the id for the yt video ")
    transcript: Optional[str] = Field(default=None, description="The transcript of the youtube video")
    summary: Optional[str] = Field(default=None, description="The summary of the youtube video transcript")
    keyword: Optional[str] = Field(default=None, description="The key word extracted from the youtube video transcript")
    video_suggestions: Optional[list[str]] = Field(default=None, description="The suggested title and description for the youtube video")
    questions: Optional[str] = Field(default=None, description="The suggested questions based on the youtube video transcript")
    next_steps: Optional[str] = Field(default=None, description="The suggested next steps based on the summary of the youtube video transcript")

class ExtractedVideoID(BaseModel):
    video_id: str = Field(description="The ID of the youtube video")

def extract_video_id(state: GraphNode):
    video_url = state.video_url
    template = PromptTemplate(
        template='''
        Extract the video ID from the following YouTube URL: {video_url}
        Return only the video ID.
        ''',
        input_variables=["video_url"]
    )
    llm_with_structured_output = llm.with_structured_output(ExtractedVideoID)
    chain = template | llm_with_structured_output
    response = chain.invoke({"video_url": video_url})
    return {"video_id": response.video_id}


def extract_transcript(state: GraphNode):
    video_id = state.video_id
    yout_api = YouTubeTranscriptApi()

    fetched_transcript = yout_api.fetch(video_id)
    transcript_text = " "

    for chars in fetched_transcript:
        transcript_text += " " + chars.text
    return {"transcript": transcript_text}


def summarize_transcript(state: GraphNode):
    transcript = state.transcript
    template = PromptTemplate(
        template='''
        Summarize the following transcript in a concise manner:
        {transcript}
        ''',
        input_variables=["transcript"]
    )
    chain = template | llm
    response = chain.invoke({"transcript": transcript})
    return {"summary": response.content}



def generate_questions(state: GraphNode):
    summary = state.summary
    template = PromptTemplate(
        template='''
        Generate 5 questions based on the following summary:
        {summary}
        ''',
        input_variables=["summary"]
    )
    chain = template | llm
    response = chain.invoke({"summary": summary})
    return {"questions": response.content}

def next_work(state: GraphNode):
    summary = state.summary
    template = PromptTemplate(
        template='''
        Based on the following summary, suggest the next steps:
        {summary}
        For example, if the video is about React basic, suggest learning about state management or hooks.
        ''',
        input_variables=["summary"]
    )
    chain = template | llm
    response = chain.invoke({"summary": summary})
    return {"next_steps": response.content}


def find_keywords(state: GraphNode):
    transcript = state.transcript
    template = PromptTemplate(
        template='''
        Extract the most relevant keyword from the following transcript:
        {transcript}
        The keyword should be a single word or a short phrase that best represents the main topic of the transcript.
        For example, if the video is about React basic, return "React".
        Return only the keyword. 
        ''',
        input_variables=["transcript"]
    )
    chain = template | llm
    response = chain.invoke({"transcript": transcript})
    return {"keyword": response.content}

def video_suggestion(state: GraphNode):
    keyword = state.keyword
    tool = YouTubeSearchTool()
    video_suggestions = tool.run(keyword)
    return {"video_suggestions": video_suggestions}



builder = StateGraph(GraphNode)

builder.add_node("extract_video_id", extract_video_id)
builder.add_node("extract_transcript", extract_transcript)
builder.add_node("summarize_transcript", summarize_transcript)
builder.add_node("generate_questions", generate_questions)  
builder.add_node("next_steps", next_work)
builder.add_node("find_keywords", find_keywords)
builder.add_node("video_suggestion", video_suggestion)

builder.add_edge(START, "extract_video_id")
builder.add_edge("extract_video_id", "extract_transcript")
builder.add_edge("extract_transcript", "summarize_transcript")
builder.add_edge("summarize_transcript", "generate_questions")
builder.add_edge("summarize_transcript", "next_steps")
builder.add_edge("extract_transcript", "find_keywords")
builder.add_edge("find_keywords", "video_suggestion")
builder.add_edge("generate_questions", END)
builder.add_edge("next_steps", END)
builder.add_edge("video_suggestion", END)

graph = builder.compile()
