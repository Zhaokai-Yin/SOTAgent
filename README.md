<div align="center">

<h1>SOTAgent</h1>

<img src="https://img.shields.io/badge/Incomplete-0b7dbe?style=plastic" alt="Static Badge" />

</div>


> [!NOTE]
> The SOTAgent has not been released yet. It is currently under development. Therefore, you can only deploy it locally on Google ADK for now. We will further optimize it and turn it into an online service later. Stay tuned!

## ❓ What can SOTAgent do?
With the rapid increase in the number of AI conferences and papers, selecting a baseline has become extremely difficult. The latest State-of-the-art (SOTA) models on our chosen benchmarks are updated periodically. After the Paperwithcode website became unusable, finding the current state-of-the-art model became very challenging, often requiring significant effort. This is highly inefficient in terms of time and manpower. To allow researchers to focus more on paper writing and coding experiments rather than struggling to find the latest SOTA model, we developed an agent that helps us find the latest SOTA model on our desired benchmark. We named it **SOTAgent**.

---

## ❗️ How to use SOTAgent(Version 1.0)?
### 1. Install the SOTAgent
You can **clone** the SOTAgent to your local environment by running the following command:  
```bash
git clone https://github.com/Zhaokai-Yin/SOTAgent.git
```
or **download** the SOTAgent from the GitHub repository and extract it to your local environment.  
### 2. Set the environment variable
**First**, you need to download a modern package manager library - `uv`. Choose one of the following three instructions depending on your Python installation method.  
```bash
pip install uv
conda install uv
mamba install uv
```
**Second**, activate the SOTAgent environment by running the following command（depend on your system(macOS/Linux/Windows)) :  
```bash
source .venv/bin/activate
```
or
```bash
.\.venv\Scripts\Activate.ps1
```

**Third**, set the environment variable by running the following command:  
```bash
pip install -r requirements.txt
```

### 3. Use the SOTAgent
**First**, you need an **_AI API KEY_** to use the SOTAgent.  By default, the program calls Gemini's API. You can GET the Gemini API KEY from [here](https://aistudio.google.com/). If you want to use the API of other AI platforms, you need to modify the code in the `agent.py` file.  

**Second**, when you have the API KEY, you need to put the API KEY into a `.env` file which should be created in the folder `.My_First_Agent`. The `.env` file should contain the following content:  
```
API_KEY=your_api_key
```
**Third**, run the following command to start the SOTAgent:  
```bash
uv run adk web
```

**Fourth**, you can see the SOTAgent is running in your browser. You can interact with the SOTAgent. The SOTAgent will respond to your input and give you the best answer.

---

>[Warning]
> The SOTAgent is still in development and may not work as expected. Please use it with caution.

## 💕 Thanks for using SOTAgent!
If you have any questions or suggestions, please feel free to contact me. My email address is [Emali](mailto:YinZhaokai2006@outlook.com).

<div align="center">
Last update: 2025.11.17
</div>


---