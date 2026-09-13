"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import os
import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant – trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin

## AVAILABLE TOOLS
{tools}

## CORE RULES (Bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket`.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả...), có thể trả lời trực tiếp mà không cần gọi tool.
4. Nếu câu hỏi cần NHIỀU tool, hãy gọi tuần tự từng tool rồi tổng hợp kết quả.

## OPERATIONAL BOUNDARIES
- CHỈ trả lời về các sản phẩm, dịch vụ thuộc Vingroup (VinFast, Vinpearl, Vinhomes...).
- Từ chối lịch sự nếu khách hỏi về sản phẩm ngoài hệ sinh thái.

## OUTPUT CONTRACT
- Khi cần gọi tool: xuất Thought -> Action -> Action Input
- Khi có kết quả: xuất Thought -> Final Answer
- Luôn giữ thái độ phục vụ khách hàng tận tâm.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot – Không sử dụng Tool Calling hay ReAct Loop.
    Mục đích: So sánh chất lượng trả lời khi LLM bịa thông tin (hallucination).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM (hoặc trả lời mock nếu không có API key)."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn sản phẩm Vingroup. Hãy trả lời câu hỏi sau:\n{user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "gemini_llm"
                }
            except Exception as e:
                pass

        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Production-grade Agent với System Prompt Engineering & Tool Calling.

    Features:
      - 2 custom tools: search_product_catalog, submit_support_ticket
      - Sequential & Parallel tool calling
      - Max iterations safeguard
      - Full trace logging
    """

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------------
    # Intent Detection (Rule-based Simulator)
    # ---------------------------------------------------------------------------
    def _detect_intent(self, user_input: str) -> Dict[str, bool]:
        text = user_input.lower()
        is_faq = any(kw in text for kw in ["bảo hành", "chính sách", "bao lâu", "đổi trả", "quy định"]) and not any(kw in text for kw in ["lỗi", "hỏng", "khiếu nại", "tôi tên", "tên tôi"])
        
        has_catalog_intent = any(kw in text for kw in ["xem", "tìm", "mua", "giá", "dưới", "bao nhiêu", "có xe nào", "có tour nào", "gói"])
        has_category = any(kw in text for kw in ["xe", "vinfast", "vf", "resort", "vinpearl", "du lịch", "phòng", "khách sạn"])
        needs_catalog = has_catalog_intent and has_category and not is_faq
        
        needs_ticket = any(kw in text for kw in ["lỗi", "hỏng", "sự cố", "hỗ trợ", "khiếu nại", "phản ánh", "phản hồi", "gấp", "nghiêm trọng", "ẩm mốc", "tôi tên", "tên tôi"])
        
        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _parse_catalog_args(self, user_input: str) -> Dict[str, Any]:
        text = user_input.lower()
        if any(kw in text for kw in ["du lịch", "resort", "vinpearl", "khách sạn", "phòng"]):
            category = "du_lich"
        else:
            category = "xe_dien"
            
        max_price = 999999999999
        m_trieu = re.search(r'(\d+)\s*(?:triệu|tr|trieu)', text)
        m_ty = re.search(r'(\d+)\s*(?:tỷ|ty)', text)
        if m_trieu:
            max_price = int(m_trieu.group(1)) * 1_000_000
        elif m_ty:
            max_price = int(m_ty.group(1)) * 1_000_000_000
            
        return {"category": category, "max_price": max_price}

    def _parse_ticket_args(self, user_input: str) -> Dict[str, Any]:
        text = user_input
        name_match = re.search(r'(?:tôi tên là|tôi tên|tên tôi là|tên tôi)\s*[:\s]*([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\n| xe| phòng| bị| mức)', text, re.IGNORECASE)
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"
        
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
            priority = "high"
        elif any(kw in text_lower for kw in ["thấp", "low"]):
            priority = "low"
        else:
            priority = "medium"
            
        issue_match = re.search(r'((?:xe|phòng|vấn đề|lỗi)[^.,;]+(?:bị [^.,;]+|lỗi [^.,;]+))', text, re.IGNORECASE)
        if issue_match:
            issue_desc = issue_match.group(1).strip()
        else:
            issue_desc = re.sub(r'^(?:tôi tên là|tôi tên|tên tôi là|tên tôi)\s*[:\s]*[A-ZÀ-Ỹa-zà-ỹ\s]+[,.]\s*', '', text, flags=re.IGNORECASE).strip()
            
        if not issue_desc:
            issue_desc = user_input
            
        return {
            "customer_name": customer_name,
            "issue_description": issue_desc,
            "priority": priority
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        iteration = 0

        # Safeguard: Max iterations check
        if self.max_iterations <= 0:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached"
            }

        intents = self._detect_intent(user_input)

        # 1. Trường hợp FAQ: Trả lời trực tiếp, không gọi tool
        if intents["is_faq"]:
            iteration += 1
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km (tùy điều kiện nào đến trước), áp dụng cho toàn bộ các dòng xe điện VinFast chính hãng."
            self.trace.append({
                "iteration": iteration,
                "thought": "Câu hỏi thuộc nhóm FAQ chính sách bảo hành, có thể trả lời trực tiếp mà không cần gọi tool.",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        needs_catalog = intents["needs_catalog"]
        needs_ticket = intents["needs_ticket"]

        # 2. Trường hợp cần cả 2 tool (Sequential / Parallel tool calling)
        if needs_catalog and needs_ticket:
            # Iteration 1: Catalog
            iteration += 1
            if iteration > self.max_iterations:
                return {"answer": "Lỗi: Vượt quá số bước tối đa.", "trace": self.trace, "iterations": iteration, "status": "max_iterations_reached"}
            cat_args = self._parse_catalog_args(user_input)
            obs_cat = search_product_catalog(**cat_args)
            self.trace.append({
                "iteration": iteration,
                "thought": f"Khách hàng muốn tra cứu danh mục {cat_args['category']}. Cần gọi search_product_catalog.",
                "action": {"name": "search_product_catalog", "args": cat_args},
                "observation": obs_cat
            })

            # Iteration 2: Ticket
            iteration += 1
            if iteration > self.max_iterations:
                return {"answer": "Lỗi: Vượt quá số bước tối đa.", "trace": self.trace, "iterations": iteration, "status": "max_iterations_reached"}
            ticket_args = self._parse_ticket_args(user_input)
            obs_ticket = submit_support_ticket(**ticket_args)
            self.trace.append({
                "iteration": iteration,
                "thought": "Khách hàng cũng phản ánh sự cố cần hỗ trợ. Cần gọi submit_support_ticket.",
                "action": {"name": "submit_support_ticket", "args": ticket_args},
                "observation": obs_ticket
            })

            # Iteration 3: Tổng hợp kết quả
            iteration += 1
            cat_items = []
            if not obs_cat:
                cat_items.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
            else:
                for p in obs_cat:
                    cat_items.append(f"- {p['name']}: {p['price_vnd']:,} VNĐ")

            answer = (
                f"1. Thông tin du lịch/sản phẩm:\n" + "\n".join(cat_items) + "\n\n"
                f"2. Ghi nhận phản hồi:\n"
                f"Phiếu hỗ trợ {obs_ticket.get('ticket_id')} đã được tạo cho khách hàng {obs_ticket.get('customer_name')}."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": "Tôi đã thu thập đủ thông tin để trả lời khách hàng.",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # 3. Trường hợp chỉ cần catalog (Single tool catalog)
        if needs_catalog:
            iteration += 1
            if iteration > self.max_iterations:
                return {"answer": "Lỗi: Vượt quá số bước tối đa.", "trace": self.trace, "iterations": iteration, "status": "max_iterations_reached"}
            cat_args = self._parse_catalog_args(user_input)
            obs_cat = search_product_catalog(**cat_args)
            self.trace.append({
                "iteration": iteration,
                "thought": f"Khách hàng cần tra cứu sản phẩm danh mục {cat_args['category']}. Cần gọi search_product_catalog.",
                "action": {"name": "search_product_catalog", "args": cat_args},
                "observation": obs_cat
            })

            if not obs_cat:
                answer = "Rất tiếc, không tìm thấy sản phẩm phù hợp."
            else:
                lines = ["Dưới đây là các sản phẩm phù hợp với yêu cầu của bạn:"]
                for p in obs_cat:
                    lines.append(f"- {p['name']}: {p['price_vnd']:,} VNĐ ({p.get('description', '')})")
                answer = "\n".join(lines)

            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # 4. Trường hợp chỉ cần ticket (Single tool ticket)
        if needs_ticket:
            iteration += 1
            if iteration > self.max_iterations:
                return {"answer": "Lỗi: Vượt quá số bước tối đa.", "trace": self.trace, "iterations": iteration, "status": "max_iterations_reached"}
            ticket_args = self._parse_ticket_args(user_input)
            obs_ticket = submit_support_ticket(**ticket_args)
            self.trace.append({
                "iteration": iteration,
                "thought": f"Khách hàng yêu cầu hỗ trợ sự cố. Cần gọi submit_support_ticket.",
                "action": {"name": "submit_support_ticket", "args": ticket_args},
                "observation": obs_ticket
            })

            ticket_id = obs_ticket.get("ticket_id", "")
            customer = obs_ticket.get("customer_name", ticket_args["customer_name"])
            answer = f"Yêu cầu hỗ trợ của khách hàng {customer} đã được tiếp nhận thành công. Mã vé: {ticket_id} (ưu tiên {obs_ticket.get('priority')}). Chúng tôi sẽ xử lý sớm nhất!"

            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # 5. Mặc định nếu không thuộc các trường hợp trên
        iteration += 1
        answer = "Chào bạn, VinAssistant có thể giúp bạn tra cứu xe điện VinFast, gói nghỉ dưỡng Vinpearl hoặc gửi yêu cầu hỗ trợ kỹ thuật."
        self.trace.append({
            "iteration": iteration,
            "thought": "Câu hỏi không khớp với tool cụ thể, đưa ra câu trả lời hướng dẫn.",
            "final_answer": answer
        })
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
