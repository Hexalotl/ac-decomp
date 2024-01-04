from __future__ import annotations
import argparse
import os
import re
from enum import Enum
from enum import Flag
from io import StringIO

class ConditionalStackEntry(Flag):
    NONE = 0
    WRITEABLE = 2
    CONDITION_MET = 4

class EvaluationTokenType(Enum):
    NONE = 0
    OPEN_PAREN = 1
    CLOSE_PAREN = 2
    UNARY_OP = 3
    BINARY_OP = 4
    LITERAL = 5
    EXPR_END = 6

class EvaluationOperatorType(Enum):
    NONE = 0
    NOT = 1
    AND = 2
    OR = 3
    EQUALS = 4
    NOT_EQUALS = 5

evaluation_token_dict = {
        "(" : [EvaluationTokenType.OPEN_PAREN, EvaluationOperatorType.NONE],
        ")" : [EvaluationTokenType.CLOSE_PAREN, EvaluationOperatorType.NONE],
        "!" : [EvaluationTokenType.UNARY_OP, EvaluationOperatorType.NOT],
        "&&" : [EvaluationTokenType.BINARY_OP, EvaluationOperatorType.AND],
        "||" : [EvaluationTokenType.BINARY_OP, EvaluationOperatorType.OR],
        "==" : [EvaluationTokenType.BINARY_OP, EvaluationOperatorType.EQUALS],
        "!=" : [EvaluationTokenType.BINARY_OP, EvaluationOperatorType.NOT_EQUALS],
    }

class EvaluationToken:
    token_type = EvaluationTokenType.NONE
    token_operator = EvaluationOperatorType.NONE
    literal_value = 0

    def __init__(self, string_reader: StringIO):
        def peek(string_reader: StringIO, peek_count: int = 1)-> str:
            pos = string_reader.tell()
            for i in range(peek_count):
                peek_char = string_reader.read(1)
            string_reader.seek(pos)

            return peek_char
        
        c = string_reader.read(1)
        if not c:
            # End of the string reached
            self.token_type = EvaluationTokenType.EXPR_END
            self.token_operator = EvaluationOperatorType.NONE
            return
        
        peek_char = peek(string_reader)
        if c not in evaluation_token_dict and peek_char:
            # Not a recognized single character operator, but might be a two character one
            # so append another character
            combined_char = c + peek_char
            if combined_char in evaluation_token_dict:
                c = combined_char
                string_reader.read(1)

        if c in evaluation_token_dict:
            # Recognized token. Check for special case !=
            if c == "!":
                peek_char = peek(string_reader)
                if peek_char == "=":
                    c += next_char
                    string_reader.read(1)

            # Assign types from the dictionary
            token_tuple = evaluation_token_dict[c]
            self.token_type = token_tuple[0]
            self.token_operator = token_tuple[1]
            return
        
        # Not a recognized token, so treat it as a literal
        while True:
            next_char = peek(string_reader)
            if not next_char or next_char in evaluation_token_dict:
                break

            peek_char = peek(string_reader, 2)
            if peek_char:
                # Check if the combination of the current and next char make a two character operator
                two_char_token = next_char + peek_char
                if two_char_token in evaluation_token_dict:
                    break
            
            # Append character and move to the next one
            c += next_char
            string_reader.read(1)
        
        # Save the generated value
        self.token_type = EvaluationTokenType.LITERAL
        self.literal_value = int(c)

class BooleanExpression:
    left_expression: BooleanExpression = None
    right_expression: BooleanExpression = None
    
    expression_operator = EvaluationOperatorType.NONE
    literal_value = 0

    def __init__(self, operator: EvaluationOperatorType, left: BooleanExpression, right: BooleanExpression, value: int) -> None:
        self.expression_operator = operator
        self.left_expression = left
        self.right_expression = right
        self.literal_value = value

    @staticmethod
    def create_and(left: BooleanExpression, right: BooleanExpression):
        return BooleanExpression(EvaluationOperatorType.AND, left, right, 0)
    
    @staticmethod
    def create_or(left: BooleanExpression, right: BooleanExpression):
        return BooleanExpression(EvaluationOperatorType.OR, left, right, 0)
    
    @staticmethod
    def create_equals(left: BooleanExpression, right: BooleanExpression):
        return BooleanExpression(EvaluationOperatorType.EQUALS, left, right, 0)
    
    @staticmethod
    def create_not_equals(left: BooleanExpression, right: BooleanExpression):
        return BooleanExpression(EvaluationOperatorType.NOT_EQUALS, left, right, 0)
    
    @staticmethod
    def create_not(child: BooleanExpression):
        return BooleanExpression(EvaluationOperatorType.NOT, child, None, 0)
    
    @staticmethod
    def create_literal(value: int):
        return BooleanExpression(EvaluationOperatorType.NONE, None, None, value)

script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(root_dir, "src")
include_dir = os.path.join(root_dir, "include")
n64sdk_dir = ""

include_pattern = re.compile(r'^#\s*include\s*[<"](.+?)[>"]$')
guard_pattern = re.compile(r"^#\s*if(?:(n)def|def)?\s*([a-zA-Z0-9_\!\(\)\&\|=\s]*)(?:.*?)(?=(?:\s?\/\*|\s?\/\/))?")
else_pattern = re.compile(r"^#\s*(?:el(?:se|s)?(?:if)?)\s?([a-zA-Z0-9_\!\(\)\&\|=\s]*)(?=(?:\s?\/\*|\s?\/\/))?")
endif_pattern = re.compile(r"^#\s*endif")
define_pattern = re.compile(r"^#\s*define\s*([a-zA-Z0-9_]*)(\([a-zA-Z0-9_,\s]*\))?\s?(.*?)(?=(?:\s?\/\*|\s?\/\/))?")
define_token_pattern = re.compile(r"((\!)?((?:defined\()?([a-zA-Z_]+[a-zA-Z0-9_]*)(?:\))?))")
define_white_space_pattern = re.compile(r"\s+")

defines = {"F3DEX_GBI_2" : 1, "_LANGUAGE_C" : 1}
quiet = False
evaluate_preprocessor_directives = False

def to_polish_notation(token_list: list[EvaluationToken]) -> list[EvaluationToken]:
    output_queue: list[EvaluationToken] = []
    stack: list[EvaluationToken] = []

    index = 0
    while index < len(token_list):
        token = token_list[index]

        match token.token_type:
            case EvaluationTokenType.LITERAL:
                output_queue.append(token)
            case EvaluationTokenType.BINARY_OP | EvaluationTokenType.UNARY_OP | EvaluationTokenType.OPEN_PAREN:
                stack.insert(0, token)
            case EvaluationTokenType.CLOSE_PAREN:
                while stack[len(stack) - 1].token_type is not EvaluationTokenType.OPEN_PAREN:
                    output_queue.append(stack.pop())
                
                stack.pop()

                if len(stack) > 0 and stack[len(stack) - 1].token_type is EvaluationTokenType.UNARY_OP:
                    output_queue.append(stack.pop())
        
        index += 1
    
    while len(stack) > 0:
        output_queue.append(stack.pop())
    
    output_queue.reverse()
    return output_queue

def make_boolean_expression(tokens: list[EvaluationToken], index: list[int]):
    curr_token = tokens[index[0]]
    
    if curr_token.token_type == EvaluationTokenType.LITERAL:
        index[0] += 1
        return BooleanExpression.create_literal(curr_token.literal_value)
    
    if curr_token.token_operator == EvaluationOperatorType.NOT:
        index[0] += 1
        operand = make_boolean_expression(tokens, index)
        return BooleanExpression.create_not(operand)
    
    if curr_token.token_operator == EvaluationOperatorType.AND:
        index[0] += 1
        left = make_boolean_expression(tokens, index)
        right = make_boolean_expression(tokens, index)
        return BooleanExpression.create_and(left, right)
    
    if curr_token.token_operator == EvaluationOperatorType.OR:
        index[0] += 1
        left = make_boolean_expression(tokens, index)
        right = make_boolean_expression(tokens, index)
        return BooleanExpression.create_or(left, right)
    
    if curr_token.token_operator == EvaluationOperatorType.EQUALS:
        index[0] += 1
        left = make_boolean_expression(tokens, index)
        right = make_boolean_expression(tokens, index)
        return BooleanExpression.create_equals(left, right)
    
    if curr_token.token_operator == EvaluationOperatorType.NOT_EQUALS:
        index[0] += 1
        left = make_boolean_expression(tokens, index)
        right = make_boolean_expression(tokens, index)
        return BooleanExpression.create_not_equals(left, right)
    
    return None

def evaluate_boolean_expression_node(node: BooleanExpression)-> int:
    if node.expression_operator == EvaluationOperatorType.NONE:
        return node.literal_value
    
    if node.expression_operator == EvaluationOperatorType.NOT:
        is_non_zero = evaluate_boolean_expression_node(node.left_expression) != 0
        if is_non_zero:
            return 0
        return 1
    if node.expression_operator == EvaluationOperatorType.OR:
        if evaluate_boolean_expression_node(node.left_expression) != 0 or evaluate_boolean_expression_node(node.right_expression) != 0:
            return 1
        return 0
    if node.expression_operator == EvaluationOperatorType.AND:
        if evaluate_boolean_expression_node(node.left_expression) != 0 and evaluate_boolean_expression_node(node.right_expression) != 0:
            return 1
        return 0
    if node.expression_operator == EvaluationOperatorType.EQUALS:
        if evaluate_boolean_expression_node(node.left_expression) == evaluate_boolean_expression_node(node.right_expression):
            return 1
        return 0
    if node.expression_operator == EvaluationOperatorType.NOT_EQUALS:
        if evaluate_boolean_expression_node(node.left_expression) != evaluate_boolean_expression_node(node.right_expression):
            return 1
        return 0
    
    return 0

def evaluate_boolean_expression(expression: str):
    # Sanitize the string
    expression = re.sub(define_white_space_pattern, '', expression)

    tokens: list[EvaluationToken] = []
    string_reader = StringIO(expression)

    # Tokenize the string
    token = None
    while True:
        token = EvaluationToken(string_reader)
        tokens.append(token)

        if token.token_type is EvaluationTokenType.EXPR_END:
            break

    # Convert to polish notation
    polish_notation = to_polish_notation(tokens)

    # Create expressions
    root_expression_node = make_boolean_expression(polish_notation, [0])
    evaluation = evaluate_boolean_expression_node(root_expression_node) != 0
    return evaluation


def evaluate_define_token(define_token_string: str) -> str:
    if not define_token_string:
        return 0
    
    define_token_match = define_token_pattern.match(define_token_string)
    if define_token_match:
        # Check if the token exists in the defined symbols dictionary
        define_token = define_token_match[4]
        if define_token in defines:
            return defines[define_token]

        return 0
    
    return 0
        
def evaluate_if_statement(if_statement: str) -> str:
    if not if_statement:
        return True
    
    # Repalce the tokens with their numerical values
    all_split = re.findall(define_token_pattern, if_statement)
    for split in all_split:
        if not split[0]:
            continue

        define_evaluation = evaluate_define_token(split[2])  
        if_statement = if_statement.replace(split[2], str(define_evaluation))

    if_statement_evaluation = evaluate_boolean_expression(if_statement)
    return if_statement_evaluation

def import_h_file(in_file: str, r_path: str) -> str:
    rel_path = os.path.join(root_dir, r_path, in_file)
    inc_path = os.path.join(include_dir, in_file)
    n64sdk_path = os.path.join(n64sdk_dir, in_file)
    if os.path.exists(rel_path):
        return import_c_file(rel_path)
    elif os.path.exists(inc_path):
        return import_c_file(inc_path)
    elif os.path.exists(n64sdk_path):
        return import_c_file(n64sdk_path)
    else:
        if not quiet:
            print("Failed to locate", in_file)
        exit(1)


def import_c_file(in_file) -> str:
    in_file = os.path.relpath(in_file, root_dir)
    out_text = ""

    # Flag for whether the "Processing File" log has ben outputted 
    process_file_log_outputted = quiet

    can_write = True
    previous_conditions_met = True
    current_condition_met = True
    conditional_stack = [ConditionalStackEntry.CONDITION_MET | ConditionalStackEntry.WRITEABLE]

    # Local function to help with writing line and outputting log
    def write_file_line(file_line) -> str:
        nonlocal can_write
        nonlocal previous_conditions_met
        nonlocal current_condition_met

        if evaluate_preprocessor_directives:
            if not can_write or not previous_conditions_met or not current_condition_met:
                return
        
        nonlocal out_text
        out_text += file_line

        nonlocal process_file_log_outputted
        if process_file_log_outputted:
            return
        
        process_file_log_outputted = True

        nonlocal in_file
        print("Processing file", in_file)
        

    with open(in_file, encoding="utf-8") as file:
        for idx, line in enumerate(file):
            # Strip the end of the line of whitespace for our regex searches
            stripped_line = line.strip()

            # Check if the current condition for the scope we're in has been met
            current_condition_met = conditional_stack[len(conditional_stack) - 1] & ConditionalStackEntry.CONDITION_MET == ConditionalStackEntry.CONDITION_MET

            # CASE 1: #include
            include_match = include_pattern.match(stripped_line)
            if include_match:
                # To avoid expanding header files that don't apply to our project such as #ifdef TARGET_PC
                # we need to see if we've met the definition requirement. Otherwise skip the include
                if can_write and previous_conditions_met and current_condition_met:
                    write_file_line(f'/* "{in_file}" line {idx} "{include_match[1]}" */\n')
                    write_file_line(import_h_file(include_match[1], os.path.dirname(in_file)))
                    write_file_line(f'/* end "{include_match[1]}" */\n')
                continue

            # CASE 2: #endif block
            endif_match = endif_pattern.match(stripped_line)
            if endif_match:
                # End reached so we can pop the stacks
                conditional_stack.pop()
                
                # Re-evaluate the flags since the stack changed
                previous_conditions_met = True
                can_write = True
                for conditional_entry in conditional_stack:
                    previous_conditions_met &= conditional_entry & ConditionalStackEntry.CONDITION_MET == ConditionalStackEntry.CONDITION_MET
                    can_write &= conditional_entry & ConditionalStackEntry.WRITEABLE == ConditionalStackEntry.WRITEABLE
                
                if not evaluate_preprocessor_directives:
                    write_file_line(line)
                continue

            # CASE 3: #if/#ifdef/#ifndef
            guard_match = guard_pattern.match(stripped_line)
            if guard_match:
                if not can_write:
                    # Earlier evaluation makes it so that we don't need to check this
                    conditional_stack.append(ConditionalStackEntry.CONDITION_MET)
                else:
                    # What definition are we checking against?
                    is_ifndef_evaluation = False if not guard_match[1] else True
                    if_statement_to_evaluate = guard_match[2]
                    
                    current_condition_met = evaluate_if_statement(if_statement_to_evaluate)
                    if is_ifndef_evaluation:
                        current_condition_met = not current_condition_met

                    if is_ifndef_evaluation and not current_condition_met and idx == 0:
                        # Current assumption is if the first line is the ifndef guard and it fails, just short-circuit early
                        break

                    entry_to_add = ConditionalStackEntry.WRITEABLE
                    if current_condition_met:
                        entry_to_add |= ConditionalStackEntry.CONDITION_MET

                    conditional_stack.append(entry_to_add)

                if not evaluate_preprocessor_directives:
                    write_file_line(line)
                continue

            # CASE 4: #else/#elif
            else_match = else_pattern.match(stripped_line)
            if else_match:
                if current_condition_met:
                    # We alread met an earlier condition so we don't want to write any more lines
                    conditional_stack[len(conditional_stack) - 1] &= ~ConditionalStackEntry.WRITEABLE
                    can_write = False
                else:
                    else_statemet_to_evaluate = else_match[1]
                    else_statement_condition_met = True
                    
                    if else_statemet_to_evaluate:
                        else_statement_condition_met = evaluate_if_statement(else_statemet_to_evaluate)

                    if else_statement_condition_met:
                        # We have fulfilled a condition
                        conditional_stack[len(conditional_stack) - 1] |= ConditionalStackEntry.CONDITION_MET

                if not evaluate_preprocessor_directives:
                    write_file_line(line)
                continue


            # Case 5: #define
            define_match = define_pattern.match(stripped_line)
            if define_match:
                # We currently only add defines if they aren't macro functions
                if not define_match[2]:
                    if not define_match[3]:
                        defines[define_match[1]] = 1
                    else:
                        defines[define_match[1]] = define_match[3]
                else:
                    # For evaluation purposes mark the macro as defined
                    # Come back later if we want to expand out macros to inline them
                    defines[define_match[1]] = 1

                write_file_line(line)
                continue

            # CASE 6: Default
            write_file_line(line)

    return out_text


def main():
    parser = argparse.ArgumentParser(description="Create a context file which can be used for decomp.me")
    parser.add_argument("c_file", help="File from which to create context")
    parser.add_argument(
        "--relative", "-r", dest="relative", help="Extract context relative to the source file", action="store_true"
    )
    parser.add_argument(
        "--n64_sdk", "-n64", dest="n64sdk", help="Path to the N64 SDK", action="store"
    )
    parser.add_argument("--quiet", "-q", dest="quiet", help="Don't output anything", action="store_true")
    parser.add_argument("--evaluate_conditionals", "-e", dest="evaluate", help="If preprocessor directives should be evaluated and stripped", action="store_true", default=False)
    parser.add_argument("--define", "-d", dest="defines", help="Add a default definition to bring in potentially excluded sections", action="append")
    args = parser.parse_args()

    global quiet
    quiet = args.quiet

    global evaluate_preprocessor_directives
    evaluate_preprocessor_directives = args.evaluate

    global n64sdk_dir
    n64sdk_dir = os.environ['N64_SDK'] if args.n64sdk is None else args.n64sdk
    n64sdk_dir = os.path.join(n64sdk_dir, "ultra/usr/include")

    global defines
    if args.defines is not None:
        for define in args.defines:
            defines[define] = 1

    content = ""
    for definition in defines:
        content += "#define " + definition + "\n"
        
    # Don't write, but do include the special metrowerks define for evaluation purposes
    defines["__MWERKS__"] = 1

    c_file = args.c_file
    content += import_c_file(c_file)
    filename = f"{c_file}.ctx" if args.relative else os.path.join(root_dir, "ctx.c")

    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


if __name__ == "__main__":
    main()
