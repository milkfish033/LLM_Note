[中文](./What's_user_prompt_and_system_prompt_cn.md) |
[English](./What's_user_prompt_and_system_prompt.md)

# Understanding User Prompt and System Prompt in Machine Learning

## 1. What Is a Prompt?
A **prompt** is the text or instruction you feed into a model to tell it what you want it to do.  

For example:  
> "Write a Python function that returns the Fibonacci sequence."

The model takes this text, converts it into tokens (numbers representing words or subwords), processes it through its neural network, and generates a response.

---

## 2. User Prompt
**User prompt** is what *you* (the end user) input directly into the model.

- It defines the **specific task or query**.
- It is **dynamic** — changes every time depending on what you ask.
- It’s usually conversational or task-oriented.

**Example:**
> "Summarize this article in 3 bullet points."

The model interprets this as the user’s immediate instruction.

---

## 3. System Prompt
**System prompt** is a hidden or higher-level instruction that sets the model’s **overall behavior, tone, or personality**.  
It’s not written by the user — it’s usually set by the developer or platform (like OpenAI, Anthropic, or a custom chatbot developer).

**Purpose:**
- Define *how* the model should respond, not *what* to respond about.
- Ensure consistency, safety, and alignment.

**Example:**
- System prompt:
> "You are a helpful, polite, and concise assistant that always provides accurate informatio"

- If the user prompt is:
> "Tell me a joke."

- The **system prompt** ensures the model replies humorously but safely, e.g.  
> "Why did the computer show up late to work? It had a hard drive!"

---

## 4. How They Work Together
When generating a response, the model combines several layers of context:

1. **System prompt** → sets behavior and constraints  
2. **Developer prompt (optional)** → adds special rules (e.g., “Answer in JSON format”)  
3. **User prompt** → defines the task or question  
4. **Conversation history** → previous messages for context

All of this text is fed into the model together as one **input sequence**.  
The model doesn’t “know” which part is which — but the structure helps guide its reasoning.

---

## 5. Example: Combined Input to a Model

- System: You are an expert math tutor who explains with examples.
- User: Explain what an eigenvalue is.

Model’s internal combined text might look like:
> "You are an expert math tutor who explains with examples.\n\nExplain what an eigenvalue is."

Result:
> "An eigenvalue measures how much a linear transformation stretches a vector.  
> For example, if A = [[2,0],[0,3]], then the eigenvalues are 2 and 3."

---

## 6. Summary Table

| Type | Who Provides It | Purpose | Example |
|------|-----------------|----------|----------|
| **System Prompt** | Developer / Platform | Defines behavior and tone | "You are a helpful assistant." |
| **User Prompt** | End user | Specifies the actual request | "Summarize this paper in one paragraph." |


