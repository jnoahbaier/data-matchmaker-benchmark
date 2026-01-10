import json
import logging
import random
import re
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SchemaEvaluator")


class Agent:
    """Green Agent that evaluates Purple Agents on schema merging tasks."""

    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main entry point for assessment requests."""
        input_text = get_message_text(message)
        logger.info("Received assessment request")

        try:
            request = json.loads(input_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse request as JSON")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: Expected JSON with 'participants' and optional 'config'"))],
                name="Error",
            )
            return

        participants = request.get("participants", {})
        config = request.get("config", {})
        difficulty = config.get("difficulty", "easy")

        purple_url = participants.get("schema_merger") or participants.get("merger")
        logger.info(f"Assessment config: difficulty={difficulty}, purple_url={purple_url}")
        if not purple_url:
            logger.error("No purple agent URL provided")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: No 'schema_merger' or 'merger' participant provided"))],
                name="Error",
            )
            return

        await updater.update_status(
            TaskState.working, new_agent_text_message("Generating test case...")
        )

        test_case = self.generate_test_case(difficulty)

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Sending task to purple agent at {purple_url}...")
        )

        task_message = json.dumps({
            "tables": test_case["tables"],
            "task": (
                "Analyze these tables and return JSON with:\n"
                "1. primary_keys: {table_name: column_name}\n"
                "2. join_columns: [[table1.col, table2.col], ...]\n"
                "3. inconsistencies: [list of naming inconsistencies found]\n"
                "4. merged_schema: {unified_table_name: [columns]}"
            )
        })

        try:
            response_text = await self.messenger.talk_to_agent(
                message=task_message,
                url=purple_url,
                new_conversation=True,
                timeout=120
            )
        except Exception as e:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error communicating with purple agent: {e}"))],
                name="Error",
            )
            return

        await updater.update_status(
            TaskState.working, new_agent_text_message("Evaluating response...")
        )

        try:
            response = json.loads(response_text)
        except json.JSONDecodeError:
            response = self._extract_json(response_text)
            if response is None:
                response = {}

        score, details = self.evaluate_response(response, test_case["ground_truth"])
        logger.info(f"Evaluation complete: score={score}/100")

        result = {
            "score": score,
            "max_score": 100,
            "difficulty": difficulty,
            "details": details,
            "test_case": {
                "tables": [t["name"] for t in test_case["tables"]],
                "ground_truth": test_case["ground_truth"]
            },
            "purple_response": response
        }

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=result))],
            name="Evaluation Result",
        )

    def generate_test_case(self, difficulty: str = "easy", seed: int = None) -> dict:
        """Generate a test case with tables and ground truth.
        
        Args:
            difficulty: One of 'easy', 'medium', 'hard'
            seed: Optional random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        if difficulty == "easy":
            cases = [self._generate_easy_case, self._generate_easy_case_variant]
        elif difficulty == "medium":
            cases = [self._generate_medium_case, self._generate_medium_case_variant]
        else:
            cases = [self._generate_hard_case, self._generate_hard_case_variant]
        
        selected = random.choice(cases)
        logger.info(f"Generated test case: {selected.__name__}")
        return selected()

    def _generate_easy_case(self) -> dict:
        """Two tables with obvious keys and one naming inconsistency."""
        return {
            "tables": [
                {
                    "name": "customers",
                    "columns": ["cust_id", "customer_name", "email"],
                    "sample_data": [
                        {"cust_id": 1, "customer_name": "Alice", "email": "alice@example.com"},
                        {"cust_id": 2, "customer_name": "Bob", "email": "bob@example.com"}
                    ]
                },
                {
                    "name": "orders",
                    "columns": ["order_id", "customer_ID", "amount", "order_date"],
                    "sample_data": [
                        {"order_id": 101, "customer_ID": 1, "amount": 99.99, "order_date": "2024-01-15"},
                        {"order_id": 102, "customer_ID": 2, "amount": 149.50, "order_date": "2024-01-16"}
                    ]
                }
            ],
            "ground_truth": {
                "primary_keys": {"customers": "cust_id", "orders": "order_id"},
                "join_columns": [["customers.cust_id", "orders.customer_ID"]],
                "inconsistencies": ["cust_id vs customer_ID (case and naming)"],
                "merged_schema": {
                    "customer_orders": ["customer_id", "customer_name", "email", "order_id", "amount", "order_date"]
                }
            }
        }

    def _generate_easy_case_variant(self) -> dict:
        """Variant: Products and inventory tables."""
        return {
            "tables": [
                {
                    "name": "products",
                    "columns": ["product_id", "name", "category"],
                    "sample_data": [
                        {"product_id": 1, "name": "Laptop", "category": "Electronics"},
                        {"product_id": 2, "name": "Chair", "category": "Furniture"}
                    ]
                },
                {
                    "name": "inventory",
                    "columns": ["inv_id", "PRODUCT_ID", "quantity", "warehouse"],
                    "sample_data": [
                        {"inv_id": 100, "PRODUCT_ID": 1, "quantity": 50, "warehouse": "A"},
                        {"inv_id": 101, "PRODUCT_ID": 2, "quantity": 30, "warehouse": "B"}
                    ]
                }
            ],
            "ground_truth": {
                "primary_keys": {"products": "product_id", "inventory": "inv_id"},
                "join_columns": [["products.product_id", "inventory.PRODUCT_ID"]],
                "inconsistencies": ["product_id vs PRODUCT_ID (case)"],
                "merged_schema": {
                    "product_inventory": ["product_id", "name", "category", "inv_id", "quantity", "warehouse"]
                }
            }
        }

    def _generate_medium_case(self) -> dict:
        """Three tables with mixed naming conventions."""
        return {
            "tables": [
                {
                    "name": "users",
                    "columns": ["user_id", "userName", "email_address"],
                    "sample_data": [
                        {"user_id": 1, "userName": "alice123", "email_address": "alice@test.com"},
                        {"user_id": 2, "userName": "bob456", "email_address": "bob@test.com"}
                    ]
                },
                {
                    "name": "products",
                    "columns": ["ProductID", "product_name", "Price"],
                    "sample_data": [
                        {"ProductID": 10, "product_name": "Widget", "Price": 29.99},
                        {"ProductID": 11, "product_name": "Gadget", "Price": 49.99}
                    ]
                },
                {
                    "name": "purchases",
                    "columns": ["purchase_id", "USER_ID", "productId", "quantity"],
                    "sample_data": [
                        {"purchase_id": 1001, "USER_ID": 1, "productId": 10, "quantity": 2},
                        {"purchase_id": 1002, "USER_ID": 2, "productId": 11, "quantity": 1}
                    ]
                }
            ],
            "ground_truth": {
                "primary_keys": {"users": "user_id", "products": "ProductID", "purchases": "purchase_id"},
                "join_columns": [
                    ["users.user_id", "purchases.USER_ID"],
                    ["products.ProductID", "purchases.productId"]
                ],
                "inconsistencies": [
                    "user_id vs USER_ID (case)",
                    "ProductID vs productId (case)",
                    "Mixed naming: snake_case, camelCase, UPPER_CASE"
                ],
                "merged_schema": {
                    "user_purchases": ["user_id", "user_name", "email", "purchase_id", "product_id", "product_name", "price", "quantity"]
                }
            }
        }

    def _generate_medium_case_variant(self) -> dict:
        """Variant: Students, courses, and enrollments."""
        return {
            "tables": [
                {
                    "name": "students",
                    "columns": ["studentId", "full_name", "major"],
                    "sample_data": [
                        {"studentId": "S001", "full_name": "Alice Smith", "major": "CS"},
                        {"studentId": "S002", "full_name": "Bob Jones", "major": "Math"}
                    ]
                },
                {
                    "name": "courses",
                    "columns": ["COURSE_ID", "courseName", "credits"],
                    "sample_data": [
                        {"COURSE_ID": "CS101", "courseName": "Intro to Programming", "credits": 3},
                        {"COURSE_ID": "MATH201", "courseName": "Calculus II", "credits": 4}
                    ]
                },
                {
                    "name": "enrollments",
                    "columns": ["enrollment_id", "student_ID", "courseId", "grade"],
                    "sample_data": [
                        {"enrollment_id": 1, "student_ID": "S001", "courseId": "CS101", "grade": "A"},
                        {"enrollment_id": 2, "student_ID": "S002", "courseId": "MATH201", "grade": "B"}
                    ]
                }
            ],
            "ground_truth": {
                "primary_keys": {"students": "studentId", "courses": "COURSE_ID", "enrollments": "enrollment_id"},
                "join_columns": [
                    ["students.studentId", "enrollments.student_ID"],
                    ["courses.COURSE_ID", "enrollments.courseId"]
                ],
                "inconsistencies": [
                    "studentId vs student_ID (case and format)",
                    "COURSE_ID vs courseId (case)",
                    "Mixed: camelCase, UPPER_CASE, snake_case"
                ],
                "merged_schema": {
                    "student_enrollments": ["student_id", "full_name", "major", "course_id", "course_name", "credits", "enrollment_id", "grade"]
                }
            }
        }

    def _generate_hard_case(self) -> dict:
        """Five tables with complex relationships."""
        return {
            "tables": [
                {
                    "name": "employees",
                    "columns": ["emp_id", "name", "dept_id", "manager_id"],
                    "sample_data": [
                        {"emp_id": 1, "name": "John", "dept_id": 100, "manager_id": None},
                        {"emp_id": 2, "name": "Jane", "dept_id": 100, "manager_id": 1}
                    ]
                },
                {
                    "name": "departments",
                    "columns": ["DeptID", "DeptName", "location_id"],
                    "sample_data": [
                        {"DeptID": 100, "DeptName": "Engineering", "location_id": 1},
                        {"DeptID": 101, "DeptName": "Sales", "location_id": 2}
                    ]
                },
                {
                    "name": "locations",
                    "columns": ["id", "city", "country"],
                    "sample_data": [
                        {"id": 1, "city": "San Francisco", "country": "USA"},
                        {"id": 2, "city": "New York", "country": "USA"}
                    ]
                },
                {
                    "name": "projects",
                    "columns": ["project_id", "projectName", "lead_emp_id", "department"],
                    "sample_data": [
                        {"project_id": "P001", "projectName": "Alpha", "lead_emp_id": 1, "department": 100}
                    ]
                },
                {
                    "name": "assignments",
                    "columns": ["assignment_id", "Employee_ID", "Project", "hours"],
                    "sample_data": [
                        {"assignment_id": 1, "Employee_ID": 1, "Project": "P001", "hours": 40}
                    ]
                }
            ],
            "ground_truth": {
                "primary_keys": {
                    "employees": "emp_id",
                    "departments": "DeptID",
                    "locations": "id",
                    "projects": "project_id",
                    "assignments": "assignment_id"
                },
                "join_columns": [
                    ["employees.dept_id", "departments.DeptID"],
                    ["departments.location_id", "locations.id"],
                    ["employees.emp_id", "projects.lead_emp_id"],
                    ["projects.department", "departments.DeptID"],
                    ["employees.emp_id", "assignments.Employee_ID"],
                    ["projects.project_id", "assignments.Project"]
                ],
                "inconsistencies": [
                    "dept_id vs DeptID vs department",
                    "emp_id vs Employee_ID vs lead_emp_id",
                    "project_id vs Project",
                    "locations.id is ambiguous"
                ],
                "merged_schema": {
                    "organization": [
                        "employee_id", "employee_name", "manager_id",
                        "department_id", "department_name",
                        "location_id", "city", "country",
                        "project_id", "project_name",
                        "assignment_id", "hours"
                    ]
                }
            }
        }

    def _generate_hard_case_variant(self) -> dict:
        """Variant: E-commerce with suppliers, products, orders, customers, reviews."""
        return {
            "tables": [
                {
                    "name": "suppliers",
                    "columns": ["supplier_id", "supplierName", "country"],
                    "sample_data": [
                        {"supplier_id": 1, "supplierName": "Acme Corp", "country": "USA"},
                        {"supplier_id": 2, "supplierName": "Global Trade", "country": "UK"}
                    ]
                },
                {
                    "name": "products",
                    "columns": ["ProductID", "product_name", "SUPPLIER_ID", "price"],
                    "sample_data": [
                        {"ProductID": 100, "product_name": "Widget", "SUPPLIER_ID": 1, "price": 29.99},
                        {"ProductID": 101, "product_name": "Gadget", "SUPPLIER_ID": 2, "price": 49.99}
                    ]
                },
                {
                    "name": "customers",
                    "columns": ["custId", "CustomerName", "email"],
                    "sample_data": [
                        {"custId": "C001", "CustomerName": "Alice", "email": "alice@test.com"},
                        {"custId": "C002", "CustomerName": "Bob", "email": "bob@test.com"}
                    ]
                },
                {
                    "name": "orders",
                    "columns": ["order_id", "customer", "product_id", "qty"],
                    "sample_data": [
                        {"order_id": 1001, "customer": "C001", "product_id": 100, "qty": 2},
                        {"order_id": 1002, "customer": "C002", "product_id": 101, "qty": 1}
                    ]
                },
                {
                    "name": "reviews",
                    "columns": ["ReviewID", "productID", "CUSTOMER_ID", "rating"],
                    "sample_data": [
                        {"ReviewID": 1, "productID": 100, "CUSTOMER_ID": "C001", "rating": 5},
                        {"ReviewID": 2, "productID": 101, "CUSTOMER_ID": "C002", "rating": 4}
                    ]
                }
            ],
            "ground_truth": {
                "primary_keys": {
                    "suppliers": "supplier_id",
                    "products": "ProductID",
                    "customers": "custId",
                    "orders": "order_id",
                    "reviews": "ReviewID"
                },
                "join_columns": [
                    ["suppliers.supplier_id", "products.SUPPLIER_ID"],
                    ["products.ProductID", "orders.product_id"],
                    ["customers.custId", "orders.customer"],
                    ["products.ProductID", "reviews.productID"],
                    ["customers.custId", "reviews.CUSTOMER_ID"]
                ],
                "inconsistencies": [
                    "supplier_id vs SUPPLIER_ID (case)",
                    "ProductID vs product_id vs productID (case and format)",
                    "custId vs customer vs CUSTOMER_ID",
                    "Mixed: camelCase, snake_case, UPPER_CASE"
                ],
                "merged_schema": {
                    "ecommerce": [
                        "supplier_id", "supplier_name", "country",
                        "product_id", "product_name", "price",
                        "customer_id", "customer_name", "email",
                        "order_id", "quantity",
                        "review_id", "rating"
                    ]
                }
            }
        }

    def evaluate_response(self, response: dict, ground_truth: dict) -> tuple:
        """Score the purple agent's response against ground truth."""
        score = 0
        details = {}

        pk_score, pk_detail = self._score_primary_keys(
            response.get("primary_keys", {}),
            ground_truth["primary_keys"]
        )
        score += pk_score
        details["primary_keys"] = {"score": pk_score, "max": 25, "detail": pk_detail}

        join_score, join_detail = self._score_join_columns(
            response.get("join_columns", []),
            ground_truth["join_columns"]
        )
        score += join_score
        details["join_columns"] = {"score": join_score, "max": 25, "detail": join_detail}

        inc_score, inc_detail = self._score_inconsistencies(
            response.get("inconsistencies", []),
            ground_truth["inconsistencies"]
        )
        score += inc_score
        details["inconsistencies"] = {"score": inc_score, "max": 25, "detail": inc_detail}

        schema_score, schema_detail = self._score_merged_schema(
            response.get("merged_schema", {}),
            ground_truth["merged_schema"]
        )
        score += schema_score
        details["merged_schema"] = {"score": schema_score, "max": 25, "detail": schema_detail}

        return score, details

    def _score_primary_keys(self, response: dict, expected: dict) -> tuple:
        if not response:
            return 0, "No primary keys provided"
        correct = 0
        total = len(expected)
        for table, key in expected.items():
            resp_key = response.get(table, "").lower().replace("_", "")
            exp_key = key.lower().replace("_", "")
            if resp_key == exp_key:
                correct += 1
        score = int(25 * correct / total) if total > 0 else 0
        return score, f"{correct}/{total} tables correct"

    def _score_join_columns(self, response: list, expected: list) -> tuple:
        if not response:
            return 0, "No join columns provided"
        def normalize_pair(pair):
            if len(pair) != 2:
                return None
            return tuple(sorted([p.lower().replace("_", "") for p in pair]))
        expected_set = {normalize_pair(p) for p in expected if normalize_pair(p)}
        response_set = {normalize_pair(p) for p in response if normalize_pair(p)}
        correct = len(expected_set & response_set)
        total = len(expected_set)
        score = int(25 * correct / total) if total > 0 else 0
        return score, f"{correct}/{total} join relationships found"

    def _score_inconsistencies(self, response: list, expected: list) -> tuple:
        """Score naming inconsistency detection with nuanced evaluation.
        
        Scoring breakdown (25 points total):
        - Count match: 5 points if detected count is within 1 of expected
        - Column mentions: 10 points based on mentioning specific columns
        - Terminology: 10 points for using proper terms (case, naming, etc.)
        """
        if not response:
            return 0, "No inconsistencies identified"
        
        response_text = " ".join(response).lower()
        total_score = 0
        
        # Count match (5 points) - reward if they found approximately right number
        expected_count = len(expected)
        actual_count = len(response)
        if actual_count == expected_count:
            total_score += 5
        elif abs(actual_count - expected_count) == 1:
            total_score += 3
        elif actual_count > 0:
            total_score += 1
        
        # Column name mentions (10 points) - check if specific columns are mentioned
        column_patterns = []
        for exp in expected:
            # Extract column names from expected (e.g., "cust_id vs customer_ID" -> ["cust_id", "customer_ID"])
            words = re.findall(r'[a-zA-Z_]+[iI][dD]|[a-zA-Z_]+_[a-zA-Z_]+', exp)
            column_patterns.extend(words)
        
        columns_mentioned = 0
        for col in column_patterns:
            # Normalize for comparison
            col_normalized = col.lower().replace("_", "")
            if col_normalized in response_text.replace("_", ""):
                columns_mentioned += 1
        
        if column_patterns:
            col_score = min(10, int(10 * columns_mentioned / len(column_patterns)))
        else:
            col_score = 5 if len(response) > 0 else 0
        total_score += col_score
        
        # Terminology (10 points) - check for proper terminology
        terminology = {
            "case": 2,           # mentions case sensitivity
            "naming": 2,         # mentions naming conventions
            "convention": 2,     # mentions conventions
            "inconsisten": 2,    # mentions inconsistency
            "snake": 1,          # mentions snake_case
            "camel": 1,          # mentions camelCase  
            "upper": 1,          # mentions UPPER_CASE
            "format": 1,         # mentions format differences
        }
        
        term_score = 0
        for term, points in terminology.items():
            if term in response_text:
                term_score += points
        total_score += min(10, term_score)
        
        return min(25, total_score), f"Found {actual_count} inconsistencies (expected {expected_count})"

    def _score_merged_schema(self, response: dict, expected: dict) -> tuple:
        if not response:
            return 0, "No merged schema provided"
        if not isinstance(response, dict):
            return 5, "Schema provided but wrong format"
        expected_cols = set()
        for cols in expected.values():
            expected_cols.update(c.lower().replace("_", "") for c in cols)
        response_cols = set()
        for cols in response.values():
            if isinstance(cols, list):
                response_cols.update(c.lower().replace("_", "") for c in cols)
        if not response_cols:
            return 5, "Schema has no columns"
        overlap = len(expected_cols & response_cols)
        coverage = overlap / len(expected_cols) if expected_cols else 0
        score = int(25 * coverage)
        return score, f"{overlap}/{len(expected_cols)} expected columns present"

    def _extract_json(self, text: str):
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return None
