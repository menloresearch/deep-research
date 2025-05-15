re_call_template_sys = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions.

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

prompt_template_dict = {}
prompt_template_dict['re_call_template_sys'] = re_call_template_sys
