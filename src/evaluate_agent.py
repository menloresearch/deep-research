#!/usr/bin/env python3
"""
Evaluate DeepResearchAgent using a simple QA evaluation framework.
"""

import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

from .agent import DeepResearchAgent
from .config import DEFAULT_LLM_MODEL, logger

# Load environment variables
load_dotenv()

class AgentEvaluator:
    """Evaluator for DeepResearchAgent responses."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.agent = DeepResearchAgent(llm_model_name=DEFAULT_LLM_MODEL)
        
    def evaluate_response(self, question: str, target_answer: str, predicted_answer: str) -> Dict[str, Any]:
        """Evaluate a single response using simple matching criteria."""
        # Convert both answers to lowercase for comparison
        target_lower = target_answer.lower()
        predicted_lower = predicted_answer.lower()
        
        # Check if target answer is contained within predicted answer
        is_correct = target_lower in predicted_lower
        
        # Calculate word overlap
        target_words = set(target_lower.split())
        predicted_words = set(predicted_lower.split())
        word_overlap = len(target_words.intersection(predicted_words))
        total_target_words = len(target_words)
        
        # Calculate overlap ratio
        overlap_ratio = word_overlap / total_target_words if total_target_words > 0 else 0
        
        return {
            "question": question,
            "target_answer": target_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "word_overlap": word_overlap,
            "total_target_words": total_target_words,
            "overlap_ratio": overlap_ratio
        }

    def evaluate_test_set(self, test_questions: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate the agent on a set of test questions."""
        results = []
        
        for test_case in test_questions[:100]:
            question = test_case["problem"]
            target_answer = test_case["answer"]
            
            # Get agent's response
            try:
                predicted_answer = self.agent.run(question)
                
                # Evaluate the response
                evaluation = self.evaluate_response(question, target_answer, predicted_answer)
                results.append(evaluation)
                
                logger.info(f"Processed question: {question[:100]}...")
                logger.info(f"Correct: {evaluation['is_correct']}, Overlap ratio: {evaluation['overlap_ratio']:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                results.append({
                    "question": question,
                    "target_answer": target_answer,
                    "predicted_answer": "ERROR",
                    "is_correct": False,
                    "word_overlap": 0,
                    "total_target_words": len(target_answer.split()),
                    "overlap_ratio": 0,
                    "error": str(e)
                })
        
        # Calculate metrics
        total = len(results)
        correct = sum(r["is_correct"] for r in results)
        avg_overlap = sum(r["overlap_ratio"] for r in results) / total if total > 0 else 0
        
        metrics = {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
            "average_overlap_ratio": avg_overlap
        }
        
        return {
            "metrics": metrics,
            "results": results
        }

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DeepResearchAgent using SimpleQA")
    parser.add_argument("--test-file", type=str, default="simple_qa_test_set.csv", help="Path to test questions CSV file")
    parser.add_argument("--output-file", type=str, default="evaluation_results.json", help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Load test questions
    try:
        test_df = pd.read_csv(args.test_file)
        test_questions = test_df.to_dict('records')
        print(f"Loaded {len(test_questions)} test questions")
    except Exception as e:
        print(f"Error loading test file: {e}")
        return
    
    # Initialize evaluator and run evaluation
    evaluator = AgentEvaluator()
    evaluation_results = evaluator.evaluate_test_set(test_questions)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    metrics = evaluation_results["metrics"]
    print("\nEvaluation Results:")
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Correct Answers: {metrics['correct_answers']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Average Word Overlap Ratio: {metrics['average_overlap_ratio']:.3f}")
    print(f"\nDetailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 