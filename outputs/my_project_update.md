# Project Update — Week 2

**Generated:** 2026-04-02 23:31:28
**Student Name:** [Your Name]

---

## Original Project Definition (Week 1)

# My Research Agent Project
**Created:** 2026-03-21 23:43:09


# MY RESEARCH AGENT PROJECT

## 1. PROJECT TITLE
[Your project name]

## 2. THE PROBLEM
[What research problem are you solving?]

## 3. YOUR SOLUTION
[How will your agent solve this?]

## 4. USER WORKFLOW
[How will someone use it?]

## 5. COMPONENTS
[Which course techniques will you use?]

☐ CO-STAR prompting - 
☐ Structured outputs - 
☐ Chain-of-thought - 
☐ Model selection - 
☐ MCP/Tool use - 
☐ Multi-step workflow - 

## 6. SUCCESS CRITERIA
[How will you measure success?]

## 7. SCOPE
IN SCOPE:
- 
- 
- 

OUT OF SCOPE:
- 
- 

## 8. DATA SOURCES
[Where does your data come from?]

## 9. TECH STACK
[What tools/libraries?]

## 10. TIMELINE
Week 1:
- 

Week 2-3:
- 

Week 4 (Insight I):
- 

Week 5-6:
- 

Week 7 (Insight II):
- 

Week 8-10:
- 

## 11. RISKS & MITIGATION
Risk 1: 
  Mitigation: 

Risk 2: 
  Mitigation: 

## 12. STRETCH GOALS
- 
- 


---

## AI Feedback

# Feedback on Your Research Agent Project

I'd love to help you refine your research agent project! However, I notice that your template is still blank - you haven't filled in the specific details yet.

## Here's what I need from you to give meaningful feedback:

**At minimum, please share:**
1. **Project title** - What are you building?
2. **The problem** - What research challenge are you addressing?
3. **Your solution approach** - How will your agent work?
4. **Which techniques** you're planning to use from the course

## In the meantime, here are some 

---

## Week 2 Updates

### Data Strategy

| Aspect | Plan |
|---|---|
| Scraping targets | [] |
| Expected volume | [e.g., 500 documents, ~200K tokens] |
| Cleaning concerns | [] |
| Tokenizer | [tiktoken cl100k_base / HF gpt2 / sentencepiece] |
| Est. cost per call | [$ per API call] |
| Approach | [fine-tuning / RAG / prompt engineering / hybrid] |

### Architecture Constraints

I notice your message contains a placeholder '[YOUR DOMAIN]' that wasn't filled in. I can't give you domain-specific advice without knowing what you're actually building.

**To get a useful answer, tell me:**

- What is your actual domain? (e.g., legal document review, genomics, customer support, code generation, medical imaging reports, financial analysis)
- What kind of inputs does your system process? (documents, conversations, structured data, code, etc.)
- What is your approximate cost budget or infrastructure constraints?
- What hardware do you have available for potential fine-tuning?

---

**Why this matters concretely:**

The answers diverge dramatically by domain. For example:

- **Legal contracts** → inputs routinely hit 50K-200K tokens, RAG is often wrong choice, hierarchical s

### Key Learnings from Week 2

- Transformer attention is O(n²) — must plan for context window limits
- Tokenizer choice affects cost: tiktoken (cl100k) is most efficient for English
- Data cleaning pipeline: language filter → dedup → PII removal
- Extended thinking improves reasoning at ~2× token cost
- Fine-tuning covered conceptually; full implementation in later class

### Updated Technical Approach

[TODO: Based on what you learned in Week 2, update your approach here.
Consider: model choice, data pipeline, tokenization, context window strategy,
and whether fine-tuning or RAG makes more sense for your use case.]

---

*Updated after completing Week 2 notebooks 00–08.*
