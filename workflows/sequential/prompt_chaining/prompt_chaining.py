from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict
from IPython.display import Image
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Step 1 :  setup of hugging face=====================================
# Load environment variables from .env
load_dotenv()
api_key = os.getenv("HF_TOKEN")
# Optional: If you want to use the raw HF InferenceClient
client = InferenceClient(
    provider="auto",
    token=api_key,
)

# Step 2: create models=================================================

# chatmodel 1
llm1 = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    task="text-generation",
)

model1 = ChatHuggingFace(llm=llm1)

# chatmodel 2
llm2 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

# Wrap as Chat Model
model2 = ChatHuggingFace(llm=llm2)


class BlogState(TypedDict):
    title: str
    outline: str
    content: str



graph = StateGraph(BlogState)

# node function
def create_outline(state: BlogState) -> BlogState:

    # fetch title
    title = state['title']

    # call llm gen outline
    prompt = f'Generate a detailed outline for a blog on the topic - {title}'
    outline = model1.invoke(prompt).content

    # update state
    state['outline'] = outline

    return state

def create_blog(state: BlogState) -> BlogState:

    title = state['title']
    outline = state['outline']

    prompt = f'Write a detailed blog on the title - {title} using the follwing outline \n {outline}'

    content = model2.invoke(prompt).content

    state['content'] = content

    return state

# nodes
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

# edges
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

workflow = graph.compile()
intial_state = {'title': 'Rise of AI in India'}
final_state = workflow.invoke(intial_state)

# print(final_state)
# {
#  'title': 'Rise of AI in India',
#  'outline': 'Blog Outline  \n“Rise of AI in India: From Policy to Practice, and What Comes Next”\n\n0.00 Meta & SEO Front-Matter  \n• Target primary keyword: “AI in India”  \n• Secondary keywords: Indian AI startups, NITI Aayog AI, AI policy India, AI talent India, AI adoption in Indian enterprises, AI ethics India.  \n• Estimated length: 3,500–4,000 words; read time 14–16 min.  \n• Suggested URL slug: /rise-of-ai-in-india-2024-guide  \n• Featured snippet angle: 120-character answer—“AI in India is surging on the back of government missions, 500+ startups and $8 B+ investments since 2018.”\n\n----------------------------------------------------\n1. Hero Section (300–350 words)  \n• Eye-catching stat animation: “India created 1 AI patent every 55 minutes in 2023”  \n• Micro-story: How a Varanasi weaver uses a computer-vision loom co-developed by IIT-BHU & TCS.  \n• Thesis statement: India is not just consuming AI; it is rapidly becoming an AI producer, regulator and exporter.\n\n2. Table of Contents (click-to-scroll)\n\n3. Executive Summary: 5 Takeaways in <120 seconds  \n• Government push via NITI Aayog & INDIAai portal  \n• 525 active AI-first startups (Tracxn, June 2024)  \n• $8.2 B VC & PE inflow, 2020-24  \n• 3.14 lakh students enrolled in AI-specific degrees or nano-courses  \n• India ranks 1st in AI skill penetration (Stanford AI Index 2024) but 40th in compute density—opportunity gap.\n\n----------------------------------------------------\n4. A Brief Historical Timeline (Interactive Scroller)  \n• 1950s–1970s: Early symbolic AI at IIT Kanpur & TIFR  \n• 1986: Fifth-Generation Computer Systems (FGCS) Indo-Japan collaboration  \n• 1991: C-DAC PARAM supercomputer—foundation for high-performance compute  \n• 2005–2015: IT services pivot to data analytics; birth of Fractal, Mu Sigma  \n• 2018: NITI Aayog releases National AI Strategy #AIforAll  \n• 2020: MeitY launches National AI Portal + Responsible AI guidelines  \n• 2023: India assumes Chair of Global Partnership on AI (GPAI)  \n• 2024: INR 10,300 Cr IndiaAI Mission approved in Union Budget\n\n----------------------------------------------------\n5. Government & Policy Ecosystem (In-depth, 500–550 words)  \n5.1 Key Organizations  \n   • NITI Aayog – strategy & sandbox programs  \n   • MeitY – policy & procurement frameworks  \n   • DPIIT – AI startup recognition & tax incentives  \n   • CDAC & STPI – compute infrastructure & incubation  \n5.2 Flagship Programs  \n   • INDIAai Knowledge Cloud, AI Research Analytics & Knowledge Dissemination Platform (AIRAWAT) super-computer (8.5 petaflops)  \n   • AI-for-All Massive Open Online Courses (17 languages)  \n   • Digital Public Goods: Bhashini (speech AI for Indian languages)  \n5.3 Regulatory Roadmap  \n   • Digital Personal Data Protection Act 2023 – impact on AI training data  \n   • Draft National Data Governance Framework Policy (open data & data trusts)  \n   • Proposed “Risk-Based AI Regulation” whitepaper (June 2024) – tiered compliance\n\n----------------------------------------------------\n6. Enterprise Adoption by Sector (Data-rich, 700–750 words)  \n6.1 BFSI  \n   • HDFC Bank: AI-driven credit underwriting reduced NPAs by 27%  \n   • NPCI: Deep-learning fraud-detection engine, 250 ms response time  \n6.2 Healthcare  \n   • Qure.ai: 7 million chest X-rays analysed across 40 countries  \n   • AIIMS-Delhi & Microsoft: rural diabetic-retinopathy screen (94% accuracy)  \n6.3 Retail & Consumer  \n   • Flipkart: multilingual conversational agent handles 80% pre-sale queries  \n   • Reliance JioMart: demand-forecasting model reduced food waste by 12%  \n6.4 Manufacturing & Industrial IoT  \n   • Tata Steel: computer-vision defect detection saves $12 M/year  \n   • Bosch India: Edge-AI energy-optimisation at Chennai plant (18% power cut)  \n6.5 Agriculture  \n   • Fasal & Intello Labs: crop-health models in 11 states, 3.2 % yield uplift  \n6.6 Government & Smart Cities  \n   • Ahmedabad traffic-management AI cuts commute time by 23%  \n   • MyGov Corona Helpdesk processed 1.3 B queries in 2021\n\n----------------------------------------------------\n7. Startup & Investment Landscape (Maps & Charts, 600–650 words)  \n• Heat-map: Top 7 hubs—Bangalore (195), Delhi-NCR (108), Mumbai (72), Hyderabad (51), Chennai (37), Pune (24), Kochi (11)  \n• Funding stages: Seed (42%), Series A (31%), Series B+ (15%), M&A (12%)  \n• Exits: 9 unicorns (Fractal, Gupshup, CitiusTech, InMobi, etc.)  \n• Government funds: INR 945 Cr Fund-of-Funds for AI/ML startups (SIDBI)  \n• Corporate VC arms: Reliance JioGenNext, TCS Co-Innovation, Wipro Ventures  \n• Case Studies (100-word capsules)  \n   1. Mad Street Den (Chennai) – retail vision AI in 20+ countries  \n   2. Yellow.ai (Bangalore) – omnichannel conversational AI, $102 M Series C  \n   3. AgNext (Mohali) – agritech spectroscopy, partnered with ITC\n\n----------------------------------------------------\n8. Talent Pipeline & Education (550–600 words)  \n8.1 Formal Education  \n   • 1,100+ engineering colleges offer AI/ML electives (AICTE model curriculum 2021)  \n   • 6 IITs now have full B.Tech in AI (IIT-Delhi, IIT-Hyderabad, etc.)  \n   • 1.5 lakh students on SWAYAM “Foundations of AI” MOOC  \n8.2 Up-skilling & Re-skilling  \n   • NASSCOM FutureSkills Prime: 4.7 lakh learners, 70% working professionals  \n   • Corporate academies: TCS NQT AI track, Infosys Springboard, Accenture AI Academy  \n   • Returnship programs for women technologists (Reboot.AI, 42% placement rate)  \n8.3 Brain-gain & Diaspora  \n   • 400+ Indian-origin AI faculty in US universities—return trend via IMPRINT & VAIBHAV fellowships  \n8.4 Skill Gaps  \n   • Only 3,200 PhD-level AI researchers (need 16,000 by 2030)  \n   • Shortage of AI hardware engineers & data-governance lawyers\n\n----------------------------------------------------\n9. Infrastructure & Compute (450–500 words)  \n• Public cloud: AWS (Mumbai & Hyderabad), Azure (Pune), GCP (Delhi, Mumbai)  \n• Sovereign cloud: MeitY’s AIRAWAT & PARAM Siddhi-AI (installed at C-DAC)  \n• Edge & 5G: Jio and Airtel rolling out 5G Edge-AI nodes in 25 cities  \n• Semiconductor: 3 proposals under India Semiconductor Mission target 28 nm AI accelerators (Tata-Powerchip, Vedanta-Foxconn, ISMC)  \n• Energy: Green data-centre push (Adani, NTPC RE) to offset AI carbon footprint\n\n----------------------------------------------------\n10. Ethical, Legal & Social Dimensions (500–550 words)  \n• Constitutional angle: Article 21 (privacy) + Article 51A(h) (scientific temper)  \n• Bias & fairness: Gender bias in Hindi-BERT (Tattle-CIS study)  \n• IP & generative AI: Delhi High Court landmark “Ankit Sahni” AI-art copyright case  \n• Deep-fakes and 2024 general elections—voluntary code by IAMAI  \n• Sector-specific ethical guidelines: RBI, SEBI, IRDAI circulars 2023–24  \n• CSR & AI for Good: Wadhwani AI’s cotton-pest early-warning system, benefiting 3 million farmers\n\n----------------------------------------------------\n11. Challenges & Bottlenecks (Bullet deep-dive, 400–450 words)  \n• Talent flight to Silicon Valley (wage differential 3.2×)  \n• Compute costs > 3× US East for on-demand GPUs due to import duties  \n• Data localisation vs. open-data dilemma  \n• Lack of vernacular datasets for Indic languages  \n• Fragmented policy between central & state governments  \n• Sustainability: India’s data-centre CO₂ emissions projected to triple by 2030\n\n----------------------------------------------------\n12. Future Roadmap 2025-2030 (Scenario Planning, 400–450 words)  \n• Conservative: India becomes top-3 AI talent exporter; $50 B services revenue by 2030  \n• Optimistic: 3 domestic GPU foundries, 10 AI unicorns/year, AI contributes 1.5 % to GDP  \n• Moon-shot: India leads global AI-for-climate research, hosts GPAI headquarters, exports AI public-goods stack to Global South  \n• Key enablers  \n   • INR 20,000 Cr AI Compute PPP grid  \n   • Unified Indic LL', 'content': 'Okay, here’s a breakdown of the blog post you requested. This is written to be informational, comprehensive and SEO-friendly. I’ll break down the layout and incorporate suggestions for improvements.\n\n**Blog Outline: “Rise of AI in India”**\n\n**I. Hero Section**\n\n* **Hooking Catch:** Start strong!\n    * A compelling animation displaying a statistic about how many AI patents India has generated. \n* **Inspiring Micro-story:** Include a short, powerful story, like the one about the Varanasi weaver. The story explains a concrete use-case with real impact, highlighting AI\'s benefit on a human level.\n* **Thesis:** Concretely announce your vision.  \n    * India is becoming a global AI powerhouse.\n\n**II. Table of Contents**\n\nMake this easy for readers to navigate:\n* Include a table with headings like: "Government & Policy," "Enterprise Adoption," "Startup Landscape," and more. Each section should have sub-headings.\n* Provide a clickable link for each section (e.g., <a href="...">Government & Policy</a>).\n\n\n**III. Executive Summary**\n\n* **In 10–12 points, highlight the key takeaways.**\n    * The government is actively promoting AI through various initiatives (NITI Aayog, INDIAAI).  \n    * **Numbers:**  525+ AI ventures, over 8 billion dollars invested. \n    *  3 million students interested in AI.\n    *  **Key Programs:**  INDIAAI knowledge cloud, government AI policy, partnerships with corporations. \n    * **Focus on Enterprise Adoption:** Show examples of AI serving various sectors (healthcare, finance, retail, etc.).\n    * **Talent Path:** Mention initiatives to upskill and reskill the workforce.\n\n\n**IV. Rising on a Bedrock of Policy to Practice**\n\n* **Interactive Timeline:** Embed a  Make this section visually appealing and engaging:\n    * Timeline in Interactive format to show AI journey in stages\n    *  Include key years, milestones, governments, and organizations. \n*  Add year milestones to encourage user engagement (eg. 2018: NITI Aayog releases National AI Strategy for India  etc.)\n\n**V. Government & Policy Ecosystem**\n\n* **NITI Aayog:**  Build a section, what are their thinking about?\n* **MeitY:** What types of interventions are they taking? \n* **DPIIT:** How are they contributing through policies and incentives? \n* **CDAC & STPI:** How are they covering infrastructure and incubation? \n* **Government Approved Policies:** Incorporate specific policy names (MNREGA, DIGITAL NEWS-Pacts etc.) \n \n**VI. Enterprise Adoption by Sector (Data-Rich)**\n\n* **Data Visualization**: Use visuals such as charts or graphs.\n    * Did they have examples in mind for the different sectors? \n    * Display data in the sections to reinforce the techniques\n\n**VII. Startup & Investment Landscape**\n\n* **Heatmap:** Visualize top AI hubs. This is crucial data!\n* **Funding Landscape:** Consider funding figures and contributions over time (seed, series A, ble, etc.) \n* **Exit Strategies:** Showcase successful exits and exits to give a potential return opportunity to the investor.\n* **Government funds:** showcase their support.\n* **Corporate VC arms:** Summarize this information\n* **Case Studies:**\n   *  Mad Street Den\n   *  Yellow.ai \n   *  AgNext. \n\n\n**VIII. Talent Pipeline & Education**\n\n* **Formal Education:** Articulate it clearly in data data, numbers, and specific programs\n* **Upskilling & Re-skilling:** Expand on this topic. \n    * Relevant programs, initiatives\n    * **Key Skill Gaps:**  Highlight them, for example, lack of kube specialization\n* **Brain Gain:**  Connect to Diaspora. How it\'s providing talent power.  \n    * **Return trend:** Indian-origin AI faculty returning to work in India \n\n\n**IX. Infrastructure & Compute:** Development in telecommunications\n* **Illustrate Infrastructure: **Provide details of public clouds (AWS, Azure, GCP) with specific tie-in to India, \n    * **Sovereign Cloud:** Tie into AIRAWAT (high-performance supercomputing) \n    * **Edge & 5G:** Highlight Jio and Airtel efforts in building the 5G Infrastructure\n    * **Semiconductor sector:** We have large semiconductor investments\n\n\n**X. Ethical, Legal & Social Dimensions**\n\n*  **Governance Strategies:** Articulate any ethical frameworks that India has implemented at the governmental level\n\n\n**XI. Challenges & Bottlenecks:** How did you provoke thought?\n\n\n**XII. Future Roadmap 2025-2030 (Scenario Planning)**\n\n* **Conservative:** How it could support India to become a top AI talent exporter by 2030?\n* **Optimistic:** How it could support India to reach a $50B AI services revenue by 2030? \n* **Moon Shot:** How it describes where India will be a world leader to study AI, especially in the advance areas of Astrobot, climate, etc.\n * **Enablers:** Good, because these cite the future! (e.g. INR 20,000 Cr AI compute grid)\n\n **Frame your blog in multiple sections.** Add sub-headings and headings with URL parameters and make a stepwise flow to guarantee that it "loads". The sections generate a strong and structured approach to your content.\n\n **Visual Enhancements**\n* **Metrics Table:**  Introduce visuals in each section - quote charts, trends that feature prominently in the Indian context, end of paragraph blocks to add them to the flow.\n   * Create a section around metrics that show the average salary for various tech jobs in India. \n\n\n**SEO Tips:**\n* **Keyword Usage:** Cluster your content around appropriate keywords (e.g. India AI policy, startup ecosystem, AI ethical guidelines). \n* **Title and Meta Description:** Invest in catchy tricks for the title and meta description to enhance SEO even more!\n\n\n**Final Thoughts:** \n\n* **Use Short Paragraphs and Scannable Bullet Points:** Increase visibility to draw more reader attention.\n* **Make it Unique and Easy to Navigate:** Add more visual appeal based upon current best practices of design.\n* **Factual Information, With the Right Metrics:** Ensure data points are accurate and current.\n* **Add Calls to Action:** (e.g., check out resource mentions).\n* **Track your KPIs:** See how your site is performing!\n\nLet me know if you\'d like to brainstorm specific points or details within these sections! \n'
#  }


# print(final_state['outline'])

# Blog Outline
# “Rise of AI in India: From Labs to Livelihoods – A Nation’s Leap Into the Intelligent Future”
#
# ------------------------------------------------
# 0. Pre-Cursor (for the writer, not the reader)
#    • Keyword cluster: AI India, AI policy 2023, Indian AI startups, AI in agriculture India, AI talent pool, NITI Aayog AI, OpenAI India, ChatGPT impact India, Make AI in India.
#    • Tone: Data-driven yet conversational, peppered with Indian idioms and pop-culture references.
#    • Goal: 3,000–3,500 words, evergreen for at least 18 months, SEO score ≥ 85.
#
# ------------------------------------------------
# 1. Hook & Scene-Setter (200–250 words)
#    • Mumbai local train, 7:30 a.m.: A vegetable vendor scrolls an AI voice app that tells him today’s optimal cauliflower price in Vashi market.
#    • Narrative pivot: “Five years ago this was sci-fi; today it’s ₹70/month SaaS.”
#    • Promise: “By the end of this post you’ll know exactly how India went from ‘AI desert’ to ‘AI force’—and what it means for your job, your startup, and your democracy.”
#
# 2. Executive Snapshot (TL;DR)
#    • Stat-card (tweetable): “India added 1 AI startup every 8 hours in 2023 → $4.1 B VC inflow → 416k AI jobs created.”
#    • One-sentence thesis: “India’s AI rise is a three-layer moat—government policy, private capital, and grassroots talent—stacked on top of a mobile-first digital public infrastructure.”
#
# 3. Historical Timeline (1990-2024) – Interactive Infographic
#    1990-2002: IIT incubation era (Raj Reddy’s speech labs).
#    2003-2010: Offshore R&D centers (Google Brain Bangalore, Yahoo Labs).
#    2011-2015: Product startups emerge (InMobi, Fractal).
#    2016-2018: #AIforAll draft, NITI Aayog paper, rise of conversational AI (Haptik).
#    2019-2021: COVID catalyst—Aarogya Setu, BlueDot India.
#    2022-2024: Gen-AI gold rush (Krutrim, Sarvam-1, Ola-Kavach LLM).
#    • Pull-quote: “We skipped the desktop decade and leapt straight to AI on the SIM card.” – Nandan Nilekani


# print(final_state['content'])

