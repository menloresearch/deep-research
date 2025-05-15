"""
Reward functions for RL training.
"""

import re

import numpy as np

from src.config import (
    BEGIN_ANSWER,
    BEGIN_SEARCH,
    BEGIN_THINK,
    END_ANSWER,
    END_SEARCH,
    END_THINK,
    INFORMATION_END,  # Added for completeness
    INFORMATION_START,  # Added for completeness, though not fully replacing variants yet
    logger,
)

# TODO: Need a reward correctness (Exact Match)


def reward_format(prompts: list, completions: list, **reward_kwargs) -> list:
    """Reward function that checks if the completion follows the required format with proper tags.

    Args:
        prompts: List of input prompts
        completions: List of completion dictionaries containing messages
        **reward_kwargs: Additional reward parameters

    Returns:
        list: List of rewards (1.0 for valid format, 0.0 for invalid)
    """
    # Regex patterns for each tag type - no markdown allowed
    think_pattern = rf"{BEGIN_THINK}[\s\S]*?{END_THINK}"
    search_pattern = rf"{BEGIN_SEARCH}[\s\S]*?{END_SEARCH}"
    answer_pattern = rf"{BEGIN_ANSWER}[\s\S]*?{END_ANSWER}"

    # Information tag patterns - handle multiple variants
    # Note: Current config has INFORMATION_START, INFORMATION_END. Variants might need separate handling or config update.
    info_patterns = [
        rf"{INFORMATION_START}[\s\S]*?{INFORMATION_END}",  # Using main config tags
        r"<info>[\s\S]*?</info>",  # Shortened - kept for now
        r"<Info[\w]*>[\s\S]*?</Info[\w]*>",  # Capitalized variants - kept for now
        r"<INFORMATION>[\s\S]*?</INFORMATION>",  # Uppercase - kept for now (could be covered by re.IGNORECASE with main tags)
        r"<INFO>[\s\S]*?</INFO>",  # Uppercase shortened - kept for now
    ]

    # Invalid patterns (bold/italic tags)
    # TODO: Consider making these configurable if they change often
    invalid_patterns = [
        r"\*\*<\/?(?:think|search|answer|information|info)>\*\*",  # Bold tags
        r"\*<\/?(?:think|search|answer|information|info)>\*",  # Italic tags
        r"_<\/?(?:think|search|answer|information|info)>_",  # Underscore italic
    ]

    rewards = []
    validation_results = {
        "has_think": [],
        "has_answer": [],
        "has_search": [],
        "has_invalid_tags": [],
        "has_info_tags": [],
        "ends_properly": [],  # New validation result
    }

    for completion in completions:
        messages = completion.get("messages", [])
        assistant_msgs = [
            msg["content"] for msg in messages if msg["role"] == "assistant"
        ]

        if not assistant_msgs:
            rewards.append(0.0)
            for key in validation_results:
                validation_results[key].append(False)
            continue

        content = assistant_msgs[-1]

        # Check if content ends with </search> or </answer> (ignoring whitespace)
        content_stripped = content.strip()
        ends_properly = content_stripped.endswith(
            END_SEARCH
        ) or content_stripped.endswith(END_ANSWER)
        validation_results["ends_properly"].append(ends_properly)

        has_invalid_tags = any(
            re.search(pattern, content) for pattern in invalid_patterns
        )
        validation_results["has_invalid_tags"].append(has_invalid_tags)
        if has_invalid_tags:
            rewards.append(0.0)
            for key in ["has_think", "has_answer", "has_search", "has_info_tags"]:
                validation_results[key].append(False)
            continue

        has_info_tags = False
        for pattern in info_patterns:
            if re.findall(pattern, content, re.IGNORECASE):
                has_info_tags = True
                break
        validation_results["has_info_tags"].append(has_info_tags)

        if has_info_tags:
            rewards.append(0.0)
            for key in ["has_think", "has_answer", "has_search"]:
                validation_results[key].append(False)
            continue

        think_matches = re.findall(think_pattern, content)
        search_matches = re.findall(search_pattern, content)
        answer_matches = re.findall(answer_pattern, content)

        has_think = len(think_matches) >= 1
        has_answer = len(answer_matches) == 1
        has_search = len(search_matches) >= 1

        validation_results["has_think"].append(has_think)
        validation_results["has_answer"].append(has_answer)
        validation_results["has_search"].append(has_search)

        if has_search and has_answer:
            rewards.append(0.0)
            continue

        # Check for proper tag sequence - think must come before answer/search
        if has_answer or has_search:
            last_think_pos = content.rfind(END_THINK)
            answer_pos = content.find(BEGIN_ANSWER) if has_answer else float("inf")
            search_pos = content.find(BEGIN_SEARCH) if has_search else float("inf")
            tag_pos = min(answer_pos, search_pos)

            if last_think_pos == -1 or last_think_pos > tag_pos:
                rewards.append(0.0)
                continue

        # Only reward if format is valid AND response ends properly
        reward = (
            1.0 if has_think and (has_answer or has_search) and ends_properly else 0.0
        )
        rewards.append(reward)

        if not reward:
            logger.debug(
                f"Format issues - think: {has_think}, answer: {has_answer}, search: {has_search}, ends_properly: {ends_properly}"
            )
            if search_matches:
                logger.debug(f"Number of search tags: {len(search_matches)}")

    logger.info(
        f"Format reward metrics - Mean: {np.mean(rewards):.3f}, Valid formats: {sum(rewards)}/{len(rewards)}"
    )
    logger.info(
        f"Responses ending properly: {sum(validation_results['ends_properly'])}/{len(rewards)}"
    )

    return rewards
