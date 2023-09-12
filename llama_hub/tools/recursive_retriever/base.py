"""Recursive Retriever tool spec."""

from llama_index.tools.tool_spec.base import BaseToolSpec
from llama_index.readers.schema.base import Document
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.indices import SummaryIndex
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.types import BaseAgent
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

from typing import Optional, Dict, List

DEFAULT_SUMMARY = """
This content contains information about {key}.
Use this index if you need to lookup specific facts about {key}.
Do not use this index if you want to analyze multiple keys.
"""

class RecursiveRetrieverToolSpec(BaseToolSpec):
    """Recursive Retriever tool spec."""

    spec_functions = ["query", "query_keys"]

    def __init__(
        self,
        agent_documents: Dict[str, List[Document]],
        agent: BaseAgent,
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        if not service_context:
            service_context = ServiceContext.from_defaults()

        agents = {}
        nodes = []

        for key, documents in agent_documents.items():
            agents[key] = self._create_subagent(documents, service_context, key, agent)
            nodes.append(IndexNode(text=DEFAULT_SUMMARY.format(key=key), index_id=key))


        vector_retriever = VectorStoreIndex(
            nodes,
            service_context=service_context
        ).as_retriever(similarity_top_k=1)

        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            query_engine_dict=agents,
            verbose=True,
        )

        response_synthesizer = get_response_synthesizer(
            service_context=service_context,
            response_mode="compact",
        )

        self.query_engine = RetrieverQueryEngine.from_args(
            recursive_retriever,
            response_synthesizer=response_synthesizer,
            service_context=service_context,
        )
        self.keys = agents.keys()

    def _create_subagent(
        self,
        documents: List[Document],
        service_context: ServiceContext,
        key: str,
        agent: BaseAgent,
    ) -> BaseAgent:
        vector_engine = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        ).as_query_engine()
        summary_engine = SummaryIndex.from_documents(
            documents, service_context=service_context
        ).as_query_engine()

        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=f"Useful for summarization questions related to {key}",
                ),
            ),
            QueryEngineTool(
                query_engine=summary_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=f"Useful for retrieving specific context from {key}",
                ),
            ),
        ]
        return agent.from_tools(
            query_engine_tools,
            llm=service_context.llm,
            verbose=True
        )

    def query(self, query: str):
        """
        Make a query to the recursive retriever to recieve information.
        Information can only be retrieved pertaining to a single key.
        Make multiple queries targetting a single key to synthesize answers to questions
        that pertain to more than one key
        """
        return self.query_engine.query(query)

    def query_keys(self):
        """
        Retrieve the list of keys corresponding to the sub-agents
        in the recursive retriever
        """
        return self.keys