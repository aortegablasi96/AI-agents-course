from agents import Runner, trace, gen_trace_id
from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
from questioning_agent import questioning_agent, FullQuery, QueryQuestionItem
from evaluator_agent import evaluator_agent, EvaluationPlan, EvalItem
from agents import Agent, function_tool
import asyncio

class ResearchManager:

    async def run(self, query: str):
        """ Run the deep research process, yielding the status updates and the final report"""
        trace_id = gen_trace_id()
        with trace("Enhanced Research trace", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"

        
            yield "Starting research..."
            full_query = await self.get_clarification_questions(query)
            yield "Clarifying questions obtained, starting the search planification..."
            search_plan = await self.plan_searches(full_query)
            yield "Searches planned, starting to search..."     
            search_results = await self.perform_searches(search_plan)
            yield "Searches complete, evaluating them..."
            evaluation_results = await self.evaluate_searches(search_results)
            
            evaluation_success = True
            for eval_item in evaluation_results.searches_and_evals:
                print(type(eval_item.eval_answer))
                print(eval_item.eval_answer)
                if not eval_item.eval_answer:
                    evaluation_success = False

            while not evaluation_success:
                yield "Evaluation not succesful, searching again to correct the wrong results..."
                new_search_plan = WebSearchPlan(searches=[])
                new_search_results = [] 
                
                for index, search_item in enumerate(search_plan):
                    if not evaluation_results.search_and_evaluations[index].eval_item:
                        new_search_plan.searches.append(search_item)
                    else:
                        new_search_results.append(search_results[index])
                    
                new_search_results.append(await self.perform_searches(new_search_plan))
                yield "New results obtained, evaluating again..."
                evaluation_results = await self.evaluate_searches(new_search_results)

                evaluation_success = True
                for eval_item in evaluation_results.searches_and_evals:
                    if not eval_item.eval_answer:
                        evaluation_success = False
            
            final_research = ""

            for item in evaluation_results.searches_and_evals:
                final_researchfinal_research += item.search

            yield "Evaluation succesful, writing report..."
            report = await self.write_report(query, final_research)
            yield "Report written, sending email..."
            await self.send_email(report)
            yield "Email sent, research complete"
            yield report.markdown_report


    async def get_clarification_questions(self, query: str) -> FullQuery:
        """ Get clarifying questions for the given query """
        print("Getting clarifying questions...")
        result = await Runner.run(
            questioning_agent,
            f"Query: {query}",
        )
        print(f"Have obtained {len(result.final_output.clarifying_questions)} clarifying questions")
        return result.final_output_as(FullQuery)

        
    async def plan_searches(self, full_query: FullQuery) -> WebSearchPlan:
        """ Plan the searches to perform for the query """
        print("Planning searches...")
        input = f"Query: {full_query.query}.\n"

        for question_item in full_query.clarifying_questions:
            input += f"Clarifying question: {question_item.question}\nReason for searching: {question_item.reason}\n"

        result = await Runner.run(
            planner_agent,
            input,
        )
        print(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output_as(WebSearchPlan)


    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """ Perform the searches to perform for the query """
        print("Searching...")
        num_completed = 0
        tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            print(f"Searching... {num_completed}/{len(tasks)} completed")
        print("Finished searching")
        return results

    async def search(self, item: WebSearchItem) -> str | None:
        """ Perform a search for the query """
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def evaluate_searches(self, search_results: list[str]) -> EvaluationPlan:
        print(f"Will perform {len(search_results)} evaluations")
        print("Evaluating...")
        num_completed = 0
        tasks = [asyncio.create_task(self.evaluate(item)) for item in search_results]
        evaluations = EvaluationPlan(searches_and_evals=[])
        for task in tasks:
            eval = await task
            if eval is not None:
                evaluations.searches_and_evals.append(eval)
            num_completed += 1
            print(f"Evaluation... {num_completed}/{len(tasks)} completed")
        print("Finished evaluating")
        return evaluations


    async def evaluate(self, search_result: str) -> EvalItem:
        """ Perform a evaluation for the search result """
        input = f"Search result: {search_result}"

        result = await Runner.run(
            evaluator_agent,
            input,
        )
        return str(result.final_output)


    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """ Write the report for the query """
        print("Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(
            writer_agent,
            input,
        )

        print("Finished writing report")
        return result.final_output_as(ReportData)

    
    async def send_email(self, report: ReportData) -> None:
        print("Writing email...")
        result = await Runner.run(
            email_agent,
            report.markdown_report,
        )
        print("Email sent")
        return report