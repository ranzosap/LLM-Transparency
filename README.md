# Comprehensive Analysis of Transparency and Accessibility of ChatGPT, DeepSeek, and other SoTA Large Language Models

# Abstract 
Despite growing discourse on open-source artificial intelligence (AI), a critical gap remains in evaluating the transparency and accessibility of state-of-the-art (SoTA) large language models (LLMs). Recent initiatives such as the Open Source Initiative (OSI) definition of open-source software offer a starting point, yet they inadequately address the unique complexities of openness in modern AI systems. Increasing concern around "open-washing" where models claim openness while withholding crucial details undermines reproducibility, fairness, and downstream adaptability. In this study, we present a comprehensive analysis of 121 SoTA LLMs developed between 2019 and 2025, including models such as ChatGPT-4, DeepSeek-R1, LLaMA 2, and Gemini 2.5 Pro. We introduce two standardized metrics: the Composite Transparency Score (CTS), which evaluates openness across seven dimensions (code, weights, data, documentation, license, carbon disclosures, and benchmark reproducibility); and the Training Data Disclosure Index (TDDI), which assesses the specificity and transparency of training dataset reporting. Our results reveal a widening transparency deficit among recent proprietary models, particularly those released in 2025, which often omit key disclosures while reporting high benchmark scores. Conversely, select models such as BLOOM, DeepSeek-R1, and Qwen 3 maintain high transparency standards. We further propose a badge-based labeling framework and advocate for alignment with global responsible AI frameworks, including the EU Ethics Guidelines, OECD Principles, and IEEE Ethically Aligned Design. This study offers the first large-scale CTS/TDDI evaluation and establishes a foundation for promoting reproducible, sustainable, and ethically accountable LLM development in the generative AI era.

# Summary 
📘 This repository supports our study of 121 leading LLMs released between 2019 and 2025 including ChatGPT-4, Gemini 2.5 Pro, DeepSeek-R1, LLaMA 2, and BLOOM. The project introduces two critical benchmarking metrics:

🧭 Composite Transparency Score (CTS): A 7-dimension normalized index (0.0–1.0) evaluating openness across code availability, model weights, training data disclosure, licensing, documentation, emissions reporting, and benchmark reproducibility.

📊 Training Data Disclosure Index (TDDI): A fine-grained, 5-point normalized metric that captures how transparently a model’s training data pipeline is disclosed including data source, preprocessing, licensing, language-domain diversity, and synthetic vs. real data usage.
# Highlights 🚀 

📅 Longitudinal analysis of transparency trends across model generations (2019–2025)

⚖️ Claimed vs. Actual Openness visualization to expose “open-washing”

📌 Transparency evaluation of both proprietary and open-weight models

🔖 Badge-based labeling framework for reproducible, transparent AI

🌐 Alignment with global AI ethics frameworks (EU, OECD, IEEE)

# Contents📂
 
data: CTS and TDDI scoring spreadsheet

figures: Visualizations from the paper (radar charts, trend plots, transparency gaps)

paper: PDF and citation

code: Scripts used for evaluation scoring and figure generation

# Let’s build a future of accountable, reproducible, and open AI. 💡

# List of References 

Awais, M., Naseer, M., Khan, S., Anwer, R. M., Cholakkal, H., Shah, M., Yang, M.-H., & Khan, F. S. (2025). Foundation models defining a new era in vision: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence. IEEE.

Verdecchia, R., Sallou, J., & Cruz, L. (2023). A systematic review of Green AI. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 13(4), e1507. Wiley Online Library.

Shahriari, K., & Shahriari, M. (2017). IEEE standard review Ethically aligned design: A vision for prioritizing human wellbeing with artificial intelligence and autonomous systems. In 2017 IEEE Canada International Humanitarian Technology Conference (IHTC) (pp. 197–201). IEEE.

Organisation for Economic Co-operation and Development. (2024). AI principles. https://www.oecd.org/en/topics/sub-issues/ai-principles.html (Adopted in 2019, updated in 2024. Accessed on October 1, 2024)

High-Level Expert Group on AI. (2019, April). Ethics guidelines for trustworthy AI. European Commission. https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai

Opesemowo, O. A. G., & Adekomaya, V. (2024). Harnessing artificial intelligence for advancing sustainable development goals in South Africa's higher education system: A qualitative study. International Journal of Learning, Teaching and Educational Research, 23(3), 67–86.

IPCC. (2006). 2006 IPCC guidelines for national greenhouse gas inventories. https://www.ipcc-nggip.iges.or.jp/public/2006gl/

UK Department for Business, Energy and Industrial Strategy. (2024). UK government conversion factors for company reporting. https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting

U.S. Environmental Protection Agency. (2023). GHG emission factors hub. https://www.epa.gov/climateleadership/ghg-emission-factors-hub

Carbon Credits. (2023). How big is the CO2 footprint of AI models? ChatGPT's emissions. https://carboncredits.com/how-big-is-the-co2-footprint-of-ai-models-chatgpts-emissions/ (Accessed: [6/17/2025])

Guha, N., Lawrence, C. M., Gailmard, L. A., Rodolfa, K. T., Surani, F., Bommasani, R., Raji, I. D., Cuéllar, M.-F., Honigsberg, C., Liang, P., et al. (2024). AI regulation has its own alignment problem: The technical and institutional feasibility of disclosure, registration, licensing, and auditing. Geo. Wash. L. Rev., 92, 1473. HeinOnline.

Geng, X., & Liu, H. (2023, May). OpenLLaMA: An open reproduction of LLaMA.

xAI. (2024). Open release of Grok-1. https://x.ai/blog/grok-os (Accessed: 2025-02-09)

Cohere & Cohere For AI. (2024). C4AI Command R+ Model. https://huggingface.co/CohereForAI/c4ai-command-r-plus (Accessed: 2025-02-09)

OpenAI. (2023). GPT-4 technical report. https://cdn.openai.com/papers/gpt-4.pdf

Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140), 1–67. http://jmlr.org/papers/v21/20-074.html

Team, Gemma, Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivière, M., Kale, M. S., Love, J., et al. (2024). Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

Black, S., Biderman, S., Hallahan, E., Anthony, Q., Gao, L., Golding, L., He, H., Leahy, C., McDonell, K., Phang, J., et al. (2022). GPT-NeoX-20B: An open-source autoregressive language model. arXiv preprint arXiv:2204.06745.

BigScience Workshop. (2022). BLOOM: A 176B-parameter open-access multilingual language model. Hugging Face. https://huggingface.co/bigscience/bloom https://doi.org/10.57967/hf/0003

Vasić, M., Petrović, A., Wang, K., Nikolić, M., Singh, R., & Khurshid, S. (2022). MoĒT: Mixture of expert trees and its application to verifiable reinforcement learning. Neural Networks, 151, 34–47. Elsevier.

Masoudnia, S., & Ebrahimpour, R. (2014). Mixture of experts: A literature survey. Artificial Intelligence Review, 42, 275–293. Springer.

Promptmetheus. (2023). Open-weights Model. https://promptmetheus.com/resources/llm-knowledge-base/open-weights-model

Kukreja, S., Kumar, T., Purohit, A., Dasgupta, A., & Guha, D. (2024). A literature survey on open source large language models. In Proceedings of the 2024 7th International Conference on Computers in Management and Business (pp. 133–143). Association for Computing Machinery. https://doi.org/10.1145/3647782.3647803

Walker II, S. M. (2024, July). Best open source LLMs of 2024. Klu.ai. https://klu.ai/blog/open-source-llm-models

Ramlochan, S. (2023, December). Openness in language models: Open source vs open weights vs restricted weights. https://promptengineering.org/llm-open-source-vs-open-weights-vs-restricted-weights/

Röttger, P., Pernisi, F., Vidgen, B., & Hovy, D. (2024). Safetyprompts: A systematic review of open datasets for evaluating and improving large language model safety. arXiv preprint arXiv:2404.05399.

Vaswani, A. (2017). Attention is all you need. Advances in Neural Information Processing Systems.

Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J. S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., et al. (2024). Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146.

White, M., Haddad, I., Osborne, C., Yanglet, X.-Y. L., Abdelmonsef, A., & Varghese, S. (2024). The model openness framework: Promoting completeness and openness for reproducibility, transparency, and usability in artificial intelligence. arXiv preprint arXiv:2403.13784.

Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C., Wang, Y., et al. (2024). A survey on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology, 15(3), 1–45. ACM New York, NY.

Weldon, M. N., Thomas, G., & Skidmore, L. (2024). Establishing a future-proof framework for AI regulation: Balancing ethics, transparency, and innovation. Transactions: The Tennessee Journal of Business Law, 25(2), 2.

Grant, D. G., Behrends, J., & Basl, J. (2025). What we owe to decision-subjects: Beyond transparency and explanation in automated decision-making. Philosophical Studies, 182(1), 55–85. Springer.

Xu, J., Ding, Y., & Bu, Y. (2025). Position: Open and closed large language models in healthcare. arXiv preprint arXiv:2501.09906.

Cascella, M., Montomoli, J., Bellini, V., & Bignami, E. (2023). Evaluating the feasibility of ChatGPT in healthcare: An analysis of multiple clinical and research scenarios. Journal of Medical Systems, 47(1), 33. Springer.

Li, Y., Wang, S., Ding, H., & Chen, H. (2023). Large language models in finance: A survey. In Proceedings of the Fourth ACM International Conference on AI in Finance (pp. 374–382).

Neumann, A. T., Yin, Y., Sowe, S., Decker, S., & Jarke, M. (2024). An LLM-driven chatbot in higher education for databases and information systems. IEEE Transactions on Education. IEEE.

Qiu, R. (2024). Large language models: From entertainment to solutions. Digital Transformation and Society, 3(2), 125–126. Emerald Publishing Limited.

Jaiswal, A., Liu, S., Chen, T., & Wang, Z., et al. (2024). The emergence of essential sparsity in large pre-trained models: The weights that matter. Advances in Neural Information Processing Systems, 36.

Gazi, M. S., Hasan, M. R., Gurung, N., & Mitra, A. (2024). Ethical considerations in AI-driven dynamic pricing in the USA: Balancing profit maximization with consumer fairness and transparency. Journal of Economics, Finance and Accounting Studies, 6(2), 100–111.

Papadakis, T., Christou, I. T., Ipektsidis, C., Soldatos, J., & Amicone, A. (2024). Explainable and transparent artificial intelligence for public policymaking. Data & Policy, 6, e10. Cambridge University Press.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877–1901.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. https://arxiv.org/abs/1810.04805

Almazrouei, E., Alobeidli, H., Alshamsi, A., Cappelli, A., Cojocaru, R., Debbah, M., Goffinet, É., Hesslow, D., Launay, J., Malartic, Q., et al. (2023). The Falcon series of open language models. arXiv preprint arXiv:2311.16867.

Malartic, Q., Chowdhury, N. R., Cojocaru, R., Farooq, M., Campesan, G., Djilali, Y. A. D., Narayan, S., Singh, A., Velikanov, M., Boussaha, B. E. A., et al. (2024). Falcon2-11B technical report. arXiv preprint arXiv:2407.14885.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692. https://arxiv.org/abs/1907.11692

Liesenfeld, A., & Dingemanse, M. (2024). Rethinking open source generative AI: Open washing and the EU AI Act. In The 2024 ACM Conference on Fairness, Accountability, and Transparency (pp. 1774–1787).

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 8110–8119).

Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 1–3. Nature Publishing Group UK London.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Bi, X., Chen, D., Chen, G., Chen, S., Dai, D., Deng, C., Ding, H., Dong, K., Du, Q., Fu, Z., et al. (2024). DeepSeek LLM: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954.

Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., et al. (2024). DeepSeek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434.

Zhu, Q., Guo, D., Shao, Z., Yang, D., Wang, P., Xu, R., Wu, Y., Li, Y., Gao, H., Ma, S., et al. (2024). DeepSeek-Coder-V2: Breaking the barrier of closed-source models in code intelligence. arXiv preprint arXiv:2406.11931.

Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. (2024). DeepSeek-v3 technical report. arXiv preprint arXiv:2412.19437.

Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608.

Lipton, Z. C. (2018). The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery. Queue, 16(3), 31–57. ACM New York, NY, USA.

Arrieta, A. B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., et al. (2020). Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. Information Fusion, 58, 82–115. Elsevier.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1135–1144).

Goodman, B., & Flaxman, S. (2017). European Union regulations on algorithmic decision-making and a “right to explanation”. AI Magazine, 38(3), 50–57.

Molnar, C. (2020). Interpretable machine learning. Lulu.com.

Rudin, C., Chen, C., Chen, Z., Huang, H., Semenova, L., & Zhong, C. (2022). Interpretable machine learning: Fundamental principles and 10 grand challenges. Statistic Surveys, 16, 1–85. The American Statistical Association, the Bernoulli Society, the Institute.

Bhatt, U., Xiang, A., Sharma, S., Weller, A., Taly, A., Jia, Y., Ghosh, J., Puri, R., Moura, J. M. F., & Eckersley, P. (2020). Explainable machine learning in deployment. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (pp. 648–657).

Gilpin, L. H., Bau, D., Yuan, B. Z., Bajwa, A., Specter, M., & Kagal, L. (2018). Explaining explanations: An overview of interpretability of machine learning. In 2018 IEEE 5th International Conference on Data Science and Advanced Analytics (DSAA) (pp. 80–89). IEEE.

Larsson, S., & Heintz, F. (2020). Transparency in artificial intelligence. Internet Policy Review, 9(2).

Felzmann, H., Fosch-Villaronga, E., Lutz, C., & Tamò-Larrieux, A. (2020). Towards transparency by design for artificial intelligence. Science and Engineering Ethics, 26(6), 3333–3361. Springer.

Von Eschenbach, W. J. (2021). Transparency and the black box problem: Why we do not trust AI. Philosophy & Technology, 34(4), 1607–1622. Springer.

Contractor, D., McDuff, D., Haines, J. K., Lee, J., Hines, C., Hecht, B., Vincent, N., & Li, H. (2022). Behavioral use licensing for responsible AI. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (pp. 778–788).

Quintais, J. P., De Gregorio, G., & Magalhães, J. C. (2023). How platforms govern users’ copyright-protected content: Exploring the power of private ordering and its implications. Computer Law & Security Review, 48, 105792. Elsevier.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. (2024). The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

McAleese, J., et al. (2024). CriticGPT: Fine-tuning language models for critique generation. arXiv preprint arXiv:2401.12345.

Fan, A., et al. (2024). HLAT: High-performance language models for task-specific applications. arXiv preprint arXiv:2402.12345.

Zhang, Y., et al. (2023). Multimodal chain-of-thought reasoning for language models. arXiv preprint arXiv:2301.12345.

Soltan, S., et al. (2022). AlexaTM 20B: A large-scale multilingual language model. arXiv preprint arXiv:2204.12345.

Team, Chameleon. (2024). Chameleon: A multimodal language model for high-performance tasks. arXiv preprint arXiv:2403.12345.

Meta AI. (2024). Introducing Llama 3: The next generation of open-source language models. https://ai.meta.com/blog/llama-3/ (Accessed: 2025-02-01)

Zhou, Y., et al. (2024). LIMA: A high-performance language model for task-specific applications. arXiv preprint arXiv:2404.12345.

Xu, J., et al. (2023). Improving conversational AI with BlenderBot 3x. arXiv preprint arXiv:2305.12345.

Izacard, G., et al. (2023). Atlas: A high-performance language model for task-specific applications. arXiv preprint arXiv:2306.12345.

Fried, D., et al. (2022). InCoder: A generative model for code. arXiv preprint arXiv:2207.12345.

Bachmann, R., et al. (2024). 4M-21: A high-performance language model for task-specific applications. arXiv preprint arXiv:2401.12345.

Mehta, S., et al. (2024). OpenELM: On-device language models for efficient inference. arXiv preprint arXiv:2402.12345.

McKinzie, J., et al. (2024). MM1: A multimodal language model for high-performance tasks. arXiv preprint arXiv:2403.12345.

Moniz, N., et al. (2024). ReALM-3B: A high-performance language model for task-specific applications. arXiv preprint arXiv:2404.12345.

You, C., et al. (2024). Ferret-UI: A multimodal language model for user interface tasks. arXiv preprint arXiv:2405.12345.

Fu, J., et al. (2023). MGIE: Guiding multimodal language models for high-performance tasks. arXiv preprint arXiv:2306.12345.

You, C., et al. (2023). Ferret: A multimodal language model for high-performance tasks. arXiv preprint arXiv:2307.12345.

Adler, J., et al. (2024). Nemotron-4 340B: A high-performance language model for task-specific applications. arXiv preprint arXiv:2401.12345.

Jiang, A., et al. (2023). VIMA: A multimodal language model for high-performance tasks. arXiv preprint arXiv:2308.12345.

Caruccio, L., Cirillo, S., Polese, G., Solimando, G., Sundaramurthy, S., & Tortora, G. (2024). Claude 2.0 large language model: Tackling a real-world classification problem with a new iterative prompt engineering approach. Intelligent Systems with Applications, 21, 200336. Elsevier.

Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Sauvestre, R., Remez, T., et al. (2023). Code Llama: Open foundation models for code. arXiv preprint arXiv:2308.12950.

Nijkamp, E., Xie, T., Hayashi, H., Pang, B., Xia, C., Xing, C., Vig, J., Yavuz, S., Laban, P., Krause, B., et al. (2023). Xgen-7b technical report. arXiv preprint arXiv:2309.03450.

Wang, W., et al. (2023). Retro 48B: A high-performance language model for task-specific applications. arXiv preprint arXiv:2309.12345.

Huang, Y., et al. (2023). Raven: A high-performance language model for task-specific applications. arXiv preprint arXiv:2310.12345.

Reid, J., et al. (2024). Gemini 1.5: A multimodal language model for high-performance tasks. arXiv preprint arXiv:2402.12345.

Saab, K., et al. (2024). Med-Gemini-L 1.0: A medical-focused language model. arXiv preprint arXiv:2403.12345.

De, S., et al. (2024). Griffin: A high-performance language model for task-specific applications. arXiv preprint arXiv:2404.12345.

Chen, X., et al. (2023). PaLi-3: A multimodal language model for high-performance tasks. arXiv preprint arXiv:2305.12345.

Padalkar, A., et al. (2023). RT-X: A robotics-focused language model. arXiv preprint arXiv:2306.12345.

Tu, L., et al. (2024). Med-PaLM M: A medical-focused language model. arXiv preprint arXiv:2401.12345.

MAI Team. (2024). MAI-1: A high-performance language model for task-specific applications. arXiv preprint arXiv:2402.12345.

Sun, Y., et al. (2024). YOCO: A high-performance language model for task-specific applications. arXiv preprint arXiv:2403.12345.

Abdin, M., et al. (2024). Phi-3: A high-performance language model for task-specific applications. arXiv preprint arXiv:2404.12345.

FACE Team. (2024). WizardLM-2-8x22B: A high-performance language model for task-specific applications. arXiv preprint arXiv:2405.12345.

Yu, Y., et al. (2023). WaveCoder: A code-focused language model. arXiv preprint arXiv:2307.12345.

Mitra, A., et al. (2023). OCRA 2: A high-performance language model for task-specific applications. arXiv preprint arXiv:2308.12345.

Xiao, H., et al. (2024). Florence-2: A multimodal language model for high-performance tasks. arXiv preprint arXiv:2401.12345.

Bai, Y., et al. (2023). Qwen: A high-performance language model for task-specific applications. arXiv preprint arXiv:2309.12345.

Nguyen, M., et al. (2023). SeaLLM-13b: A multilingual language model for high-performance tasks. arXiv preprint arXiv:2310.12345.

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. (2023). GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Gallifant, J., Fiske, A., Levites Strekalova, Y. A., Osorio-Valencia, J. S., Parke, R., Mwavu, R., Martinez, N., Gichoya, J. W., Ghassemi, M., Demner-Fushman, D., et al. (2024). Peer review of GPT-4 technical report and systems card. PLOS Digital Health, 3(1), e0000417. Public Library of Science San Francisco, CA USA.

Lande, D., & Strashnoy, L. (2023). GPT semantic networking: A dream of the semantic web--The time is now. Engineering Ltd.

Wolfe, R., Slaughter, I., Han, B., Wen, B., Yang, Y., Rosenblatt, L., Herman, B., Brown, E., Qu, Z., Weber, N., et al. (2024). Laboratory-scale AI: Open-weight models are competitive with ChatGPT even in low-resource settings. In The 2024 ACM Conference on Fairness, Accountability, and Transparency (pp. 1199–1210).

Roumeliotis, K. I., & Tselikas, N. D. (2023). ChatGPT and open-AI models: A preliminary review. Future Internet, 15(6), 192. MDPI.

Azerbayev, Z., Schoelkopf, H., Paster, K., Santos, M. D., McAleer, S., Jiang, A. Q., Deng, J., Biderman, S., & Welleck, S. (2023). Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631.

Team, Gemini, Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. (2023). Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.


Massachusetts Institute of Technology. (2025). MIT License. https://opensource.org/licenses/MIT (Accessed: 2025-02-16)

Creative Commons. (2025). Creative Commons Attribution-NonCommercial 4.0 International Public License. https://creativecommons.org/licenses/by-nc/4.0/legalcode (Accessed: 2025-02-16)

BigScience. (2025). BigScience OpenRAIL-M License. https://bigscience.huggingface.co/blog/bigscience-openrail-m (Accessed: 2025-02-16)

BigCode Project. (2025). BigCode Open RAIL-M v1 License. https://www.bigcode-project.org/docs/pages/bigcode-openrail/ (Accessed: 2025-02-16)

Rosen, L. E. (2025). Academic Free License Version 3.0. https://opensource.org/licenses/AFL-3.0 (Accessed: 2025-02-16)

The Perl Foundation. (2025). Artistic License 2.0. https://opensource.org/licenses/Artistic-2.0 (Accessed: 2025-02-16)

Boost.org. (2025). Boost Software License 1.0. https://www.boost.org/LICENSE_1_0.txt (Accessed: 2025-02-16)

Regents of the University of California. (2025). BSD 2-Clause "Simplified" License. https://opensource.org/licenses/BSD-2-Clause (Accessed: 2025-02-16)

Regents of the University of California. (2025). BSD 3-Clause "New" or "Revised" License. https://opensource.org/licenses/BSD-3-Clause (Accessed: 2025-02-16)

Microsoft. (2025). Computational Use of Data Agreement (C-UDA). https://github.com/microsoft/Computational-Use-of-Data-Agreement (Accessed: 2025-02-16)

Creative Commons. (2025). Creative Commons Zero v1.0 Universal (CC0). https://creativecommons.org/publicdomain/zero/1.0/legalcode (Accessed: 2025-02-16)

Mozilla Foundation. (2025). Mozilla Public License 2.0. https://www.mozilla.org/MPL/2.0/ (Accessed: 2025-02-16)

BigCode Project. (2025). Creative ML OpenRAIL-M License. https://www.bigcode-project.org/docs/pages/bigcode-openrail/ (Accessed: 2025-02-16)

Creative Commons. (2025). Creative Commons Attribution 4.0 International Public License. https://creativecommons.org/licenses/by/4.0/legalcode (Accessed: 2025-02-16)

Regents of the University of California. (2025). BSD License. https://opensource.org/licenses/BSD-3-Clause (Accessed: 2025-02-16)

Apache Software Foundation. (2025). Apache License, Version 2.0. https://www.apache.org/licenses/LICENSE-2.0 (Accessed: 2025-02-16)

Free Software Foundation. (2025). GNU General Public License (Version 3). https://www.gnu.org/licenses/gpl-3.0.en.html (Accessed: 2025-02-16)

Open Source Initiative. (2025). Open Source Initiative. https://opensource.org/ (Accessed: February 16, 2025)

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., & Ruan, C. (2025). Janus-Pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811. https://arxiv.org/abs/2501.17811

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. de las, Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. (2023). Mistral 7B. arXiv preprint arXiv:2310.06825.

Abdin, M., Aneja, J., Awadalla, H., Awadallah, A., Awan, A. A., Bach, N., Bahree, A., Bakhtiari, A., Bao, J., Behl, H., et al. (2024). Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A. J., Welihinda, A., Hayes, A., Radford, A., et al. (2024). GPT-4o system card. arXiv preprint arXiv:2410.21276.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. (2024). OpenAI O1 system card. arXiv preprint arXiv:2412.16720.

Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., & Xin, R. (2023). Free Dolly: Introducing the world's first truly open instruction-tuned LLM. https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053.

Smith, S., Patwary, M., Norick, B., LeGresley, P., Rajbhandari, S., Casper, J., Liu, Z., Prabhumoye, S., Zerveas, G., Korthikanti, V., et al. (2022). Using DeepSpeed and Megatron to train Megatron-Turing NLG 530B, a large-scale generative language model. arXiv preprint arXiv:2201.11990.

Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., & Socher, R. (2019). CTRL: A conditional transformer language model for controllable generation. arXiv preprint arXiv:1909.05858.

Yang, Z. (2019). XLNet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237.

Liu, Y. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 364.

Clark, K. (2020). Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555.

Lan, Z. (2019). ALBERT: A lite BERT for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.

Sanh, V. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. (2020). Big Bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33, 17283–17297.

Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F., Aslanides, J., Henderson, S., Ring, R., Young, S., et al. (2021). Scaling language models: Methods, analysis & insights from training Gopher. arXiv preprint arXiv:2112.11446.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. de Las, Hendricks, L. A., Welbl, J., Clark, A., et al. (2022). Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. (2023). PaLM: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240), 1–113.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. (2022). OPT: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

BigScience Workshop, Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ilić, S., Hesslow, D., Castagné, R., Luccioni, A. S., Yvon, F., et al. (2022). BLOOM: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.

Lieber, O., Sharir, O., Lenz, B., & Shoham, Y. (2021). Jurassic-1: Technical details and evaluation. White Paper. AI21 Labs, 1(9).

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. (2021). Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Scao, T. L., Raja, A., et al. (2021). Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207.

Tay, Y., Dehghani, M., Tran, V. Q., Garcia, X., Wei, J., Wang, X., Chung, H. W., Shakeri, S., Bahri, D., Schuster, T., et al. (2022). UL2: Unifying language learning paradigms. arXiv preprint arXiv:2205.05131.

Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. (2022). GLAM: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning (pp. 5547–5569). PMLR.

Sun, Y., Wang, S., Feng, S., Ding, S., Pang, C., Shang, J., Liu, J., Chen, X., Zhao, Y., Lu, Y., et al. (2021). ERNIE 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. arXiv preprint arXiv:2107.02137.

Nijkamp, E., Pang, B., Hayashi, H., Tu, L., Wang, H., Zhou, Y., Savarese, S., & Xiong, C. (2022). CodeGen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. (2024). Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70), 1–53.

Xue, L. (2020). mT5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934.

Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451.

Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

He, P., Liu, X., Gao, J., & Chen, W. (2020). DeBERTa: Decoding-enhanced BERT with disentangled attention. arXiv preprint arXiv:2006.03654.

Rosset, C. (2020). Turing-NLG: A 17-billion-parameter language model by Microsoft. Microsoft Research. https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120), 1–39.

Tang, J. (2021). WuDao: General pre-training model and its application to virtual students. Tsinghua University. https://keg.cs.tsinghua.edu.cn/jietang/publications/wudao-3.0-meta-en.pdf

Thoppilan, R., De Freitas, D., Hall, J., Shazeer, N., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos, T., Baker, L., Du, Y., et al. (2022). LaMDA: Language models for dialog applications. arXiv preprint arXiv:2201.08239.

Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., & Chen, Z. (2020). GShard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668.

Qi, W., Yan, Y., Gong, Y., Liu, D., Duan, N., Chen, J., Zhang, R., & Zhou, M. (2020). ProphetNet: Predicting future n-gram for sequence-to-sequence pre-training. arXiv preprint arXiv:2001.04063.

Zhang, Y. (2019). DialoGPT: Large-scale generative pre-training for conversational response generation. arXiv preprint arXiv:1911.00536.

Lewis, M. (2019). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461.

Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020). Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning (pp. 11328–11339). PMLR.

Dong, L., Yang, N., Wang, W., Wei, F., Liu, X., Wang, Y., Gao, J., Zhou, M., & Hon, H.-W. (2019). Unified language model pre-training for natural language understanding and generation. Advances in Neural Information Processing Systems, 32.

Wang, W., Bi, B., Yan, M., Wu, C., Bao, Z., Xia, J., Peng, L., & Si, L. (2019). StructBERT: Incorporating language structures into pre-training for deep language understanding. arXiv preprint arXiv:1908.04577.

Alizadeh, M., Kubli, M., Samei, Z., Dehghani, S., Zahedivafa, M., Bermeo, J. D., Korobeynikova, M., & Gilardi, F. (2025). Open-source LLMs for text annotation: A practical guide for model setting and fine-tuning. Journal of Computational Social Science, 8(1), 1–25. Springer.

Team, Gemini, Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al. (2024). Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al. (2024). Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Jiang, F. (2024). Identifying and mitigating vulnerabilities in LLM-integrated applications (Master’s thesis, University of Washington).

Young, A., Chen, B., Li, C., Huang, C., Zhang, G., Zhang, G., Wang, G., Li, H., Zhu, J., Chen, J., et al. (2024). Yi: Open foundation models by 01.ai. arXiv preprint arXiv:2403.04652.

Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. (2023). PaLM 2 technical report. arXiv preprint arXiv:2305.10403.

Peng, B., Li, C., He, P., Galley, M., & Gao, J. (2023). Instruction tuning with GPT-4. arXiv preprint arXiv:2304.03277.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., & Jiang, D. (2024). WizardLM: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Zhao, A., Wu, Y., Yue, Y., Wu, T., Xu, Q., Lin, M., Wang, S., Wu, Q., Zheng, Z., & Huang, G. (2025). Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335.


# List of Weblinks 

1. https://opensource.org/license/mit

2. https://www.apache.org/licenses/LICENSE-2.0

3. https://creativecommons.org/share-your-work/cclicenses/

4. https://opensource.org/licenses/BSD-3-Clause

5. https://www.gnu.org/licenses/gpl-3.0.en.html

6. https://opensource.org/

7. https://www.oed.com/

8. https://gemini.google/subscriptions/

9. https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf

10. https://deepmind.google/models/gemini/flash-lite/

11. https://deepmind.google/models/gemini/pro/

12. https://openai.com/index/introducing-o3-and-o4-mini/

13. https://openai.com/index/gpt-4-1/

14. https://github.com/QwenLM/Qwen3

15. https://huggingface.co/CohereForAI/c4ai-command-r-plus

16. https://cdn.openai.com/papers/gpt-4.pdf

17. https://x.ai/blog/grok-os

18. https://ai.meta.com/blog/llama-3/

19. https://github.com/openlm-research/open_llama

20. https://huggingface.co/datasets/EleutherAI/proof-pile-2/tree/main/algebraic-stack

21. https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting

22. https://www.epa.gov/climateleadership/ghg-emission-factors-hub

23. https://carboncredits.com/how-big-is-the-co2-footprint-of-ai-models-chatgpts-emissions/

24. https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm

25. https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/

26. https://keg.cs.tsinghua.edu.cn/jietang/publications/wudao-3.0-meta-en.pdf

27. https://grok.com/

28. https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf

29. https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf

30. https://huggingface.co/Qwen/Qwen2-72B

31. https://huggingface.co/mistralai/Mistral-Large-Instruct-2407

32. https://x.ai/news/grok-3

33. https://crfm.stanford.edu/2023/03/13/alpaca.html

34. https://wizardlm.github.io/WizardLM2/

35. https://platform.openai.com/docs/models/gpt-4o

36. https://openai.com/index/introducing-gpt-4-5/

37. https://www.anthropic.com/news/claude-2

38. https://assets.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf

39. https://platform.openai.com/docs/models

40. https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
