# 🎬 Video World Models for Robotics and Embodied AI

A curated reading list of papers on video world models and their applications to robotics.

Maintained as a living document. PRs welcome.

---

## 📋 Table of Contents

- [📖 Surveys](#-surveys)
- [🎥 Foundations: Video Generation](#-foundations-video-generation)
  - [🧪 Early Methods (GANs, VAEs, Flows)](#-early-methods-gans-vaes-flows)
  - [💨 Diffusion-Based Video Generation](#-diffusion-based-video-generation)
  - [⚡ Autoregressive & Transformer-Based](#-autoregressive--transformer-based)
  - [🎞️ Long Video Generation](#-long-video-generation)
  - [🎛️ Controllable Video Generation](#-controllable-video-generation)
- [🌍 World Models](#-world-models)
  - [🏛️ Foundational World Models](#-foundational-world-models)
  - [🧠 Latent-Space World Models (for RL)](#-latent-space-world-models-for-rl)
  - [🔮 Video Diffusion as World Model](#-video-diffusion-as-world-model)
  - [🌐 3D/4D World Models](#-3d4d-world-models)
  - [🎮 Interactive & Game World Models](#-interactive--game-world-models)
- [🤖 Robotics Applications](#-robotics-applications)
  - [🎓 Robot Policy Learning from Video](#-robot-policy-learning-from-video)
  - [📋 Video as Robot Action Planner](#-video-as-robot-action-planner)
  - [🏆 Reward Learning from Video](#-reward-learning-from-video)
  - [✅ Policy Evaluation with World Models](#-policy-evaluation-with-world-models)
  - [💬 Vision-Language-Action Models](#-vision-language-action-models)
- [🚗 Autonomous Driving World Models](#-autonomous-driving-world-models)
- [🔍 Self-Supervised Video Representations](#-self-supervised-video-representations)
- [🤲 Affordances from Video](#-affordances-from-video)
- [⚗️ Physical Understanding & Benchmarks](#-physical-understanding--benchmarks)
  - [⚛️ Physics in Video Models](#-physics-in-video-models)
  - [📏 Video Generation Benchmarks](#-video-generation-benchmarks)
  - [💾 Robotic Datasets](#-robotic-datasets)
- [❓ Uncertainty in World Models](#-uncertainty-in-world-models)
- [🛡️ Safety](#-safety)

---

## 📖 Surveys

**A Comprehensive Survey on World Models for Embodied AI** (2025) [[Paper]](https://arxiv.org/abs/2510.16732)  
**Understanding World or Predicting Future? A Comprehensive Survey of World Models** (2025) [[Paper]](https://arxiv.org/abs/2411.14499) — *ACM Computing Surveys*  
**Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond** (2024) [[Paper]](https://arxiv.org/abs/2405.03520)  
**A Survey of World Models for Autonomous Driving** (2025) [[Paper]](https://arxiv.org/abs/2501.11260)  
**World Models for Autonomous Driving: An Initial Survey** (2024) [[Paper]](https://arxiv.org/abs/2403.02622) — *IEEE Transactions on Intelligent Vehicles*  
**A Survey on Video Diffusion Models** (2024) [[Paper]](https://arxiv.org/abs/2405.03150) — *ACM Computing Surveys*  
**Video Diffusion Models: A Survey** (2024) [[Paper]](https://arxiv.org/abs/2405.03150)  
**Controllable Video Generation: A Survey** (2025) [[Paper]](https://arxiv.org/abs/2507.16869)  
**Generative Artificial Intelligence in Robotic Manipulation: A Survey** (2025) [[Paper]](https://arxiv.org/abs/2503.03464)  
**A Survey of Interactive Generative Video** (2025) [[Paper]](https://arxiv.org/abs/2504.21853)  
**3D and 4D World Modeling: A Survey** (2025) [[Paper]](https://arxiv.org/abs/2509.07996)  
**A Survey: Learning Embodied Intelligence from Physical Simulators and World Models** (2025) [[Paper]](https://arxiv.org/abs/2507.00917)  
**Exploring the Evolution of Physics Cognition in Video Generation: A Survey** (2025) [[Paper]](https://arxiv.org/abs/2503.21765)  
**Simulating the Visual World with Artificial Intelligence: A Roadmap** (2025) [[Paper]](https://arxiv.org/abs/2511.08585)  
**Critiques of World Models** (2025) [[Paper]](https://arxiv.org/abs/2507.05169)

---

## 🎥 Foundations: Video Generation

### 🧪 Early Methods (GANs, VAEs, Flows)

**Generative Adversarial Nets** (2014) [[Paper]](https://arxiv.org/abs/1406.2661)  
**Auto-Encoding Variational Bayes** (2013) [[Paper]](https://arxiv.org/abs/1312.6114)  
**Unsupervised Learning for Physical Interaction through Video Prediction** (2016) [[Paper]](https://arxiv.org/abs/1605.07157)  
**Generating Videos with Scene Dynamics** (2016) [[Paper]](https://arxiv.org/abs/1609.02612)  
**Dynamic Filter Networks** (2016) [[Paper]](https://arxiv.org/abs/1605.09673)  
**Stochastic Variational Video Prediction** (2017) [[Paper]](https://arxiv.org/abs/1710.11252)  
**Stochastic Adversarial Video Prediction** (2018) [[Paper]](https://arxiv.org/abs/1804.01523)  
**Adversarial Video Generation on Complex Datasets** (2019) [[Paper]](https://arxiv.org/abs/1907.06571)  
**Scaling Autoregressive Video Models** (2019) [[Paper]](https://arxiv.org/abs/1906.02634)  
**VideoFlow: A Flow-Based Generative Model for Video** (2019) [[Paper]](https://arxiv.org/abs/1903.01434)  
**Neural Discrete Representation Learning (VQ-VAE)** (2017) [[Paper]](https://arxiv.org/abs/1711.00937)  
**Predicting Video with VQVAE** (2021) [[Paper]](https://arxiv.org/abs/2103.01950)  
**FitVid: Overfitting in Pixel-Level Video Prediction** (2021) [[Paper]](https://arxiv.org/abs/2106.13195)

### 💨 Diffusion-Based Video Generation

**Denoising Diffusion Probabilistic Models** (2020) [[Paper]](https://arxiv.org/abs/2006.11239)  
**Deep Unsupervised Learning Using Nonequilibrium Thermodynamics** (2015) [[Paper]](https://arxiv.org/abs/1503.03585)  
**Diffusion Models Beat GANs on Image Synthesis** (2021) [[Paper]](https://arxiv.org/abs/2105.05233)  
**Classifier-Free Diffusion Guidance** (2022) [[Paper]](https://arxiv.org/abs/2207.12598)  
**Denoising Diffusion Implicit Models** (2022) [[Paper]](https://arxiv.org/abs/2010.02502)  
**Progressive Distillation for Fast Sampling of Diffusion Models** (2022) [[Paper]](https://arxiv.org/abs/2202.00512)  
**High-Resolution Image Synthesis with Latent Diffusion Models** (2022) [[Paper]](https://arxiv.org/abs/2112.10752)  
**Video Diffusion Models** (2022) [[Paper]](https://arxiv.org/abs/2204.03458) — *NeurIPS*  
**Imagen Video: High Definition Video Generation with Diffusion Models** (2022) [[Paper]](https://arxiv.org/abs/2210.02303)  
**MagicVideo: Efficient Video Generation with Latent Diffusion Models** (2022) [[Paper]](https://arxiv.org/abs/2211.11018)  
**Latent Video Diffusion Models for High-Fidelity Long Video Generation** (2022) [[Paper]](https://arxiv.org/abs/2211.13221)  
**VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation** (2023) [[Paper]](https://arxiv.org/abs/2303.08320)  
**Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models** (2023) [[Paper]](https://arxiv.org/abs/2304.08818)  
**ModelScope Text-to-Video Technical Report** (2023) [[Paper]](https://arxiv.org/abs/2308.06571)  
**Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets** (2023) [[Paper]](https://arxiv.org/abs/2311.15127)  
**Lumiere: A Space-Time Diffusion Model for Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2401.12945) — *SIGGRAPH Asia*  
**Latte: Latent Diffusion Transformer for Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2401.03048)  
**AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning** (2024) [[Paper]](https://arxiv.org/abs/2307.04725)  
**Scalable Diffusion Models with Transformers (DiT)** (2023) [[Paper]](https://arxiv.org/abs/2212.09748) — *ICCV*  
**CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer** (2024) [[Paper]](https://arxiv.org/abs/2408.06072)  
**HunyuanVideo: A Systematic Framework for Large Video Generative Models** (2024) [[Paper]](https://arxiv.org/abs/2412.03603)  
**Pyramidal Flow Matching for Efficient Video Generative Modeling** (2024) [[Paper]](https://arxiv.org/abs/2410.05954)  
**LTX-Video: Realtime Video Latent Diffusion** (2024) [[Paper]](https://arxiv.org/abs/2501.00103)  
**VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models** (2024) [[Paper]](https://arxiv.org/abs/2401.09047) — *CVPR*  
**Open-Sora: Democratizing Efficient Video Production for All** (2024) [[Paper]](https://arxiv.org/abs/2412.20404)  
**Open-Sora Plan: Open-Source Large Video Generation Model** (2024) [[Paper]](https://arxiv.org/abs/2412.00131)  
**Open-Sora 2.0: Training a Commercial-Level Video Generation Model in \$200K** (2025) [[Paper]](https://arxiv.org/abs/2503.09642)  
**Wan: Open and Advanced Large-Scale Video Generative Models** (2025) [[Paper]](https://arxiv.org/abs/2503.20314)  
**Flow Matching for Generative Modeling** (2023) [[Paper]](https://arxiv.org/abs/2210.02747)  
**From Slow Bidirectional to Fast Autoregressive Video Diffusion Models** (2025) [[Paper]](https://arxiv.org/abs/2412.04840) — *CVPR*  
**Consistency Models** (2023) [[Paper]](https://arxiv.org/abs/2303.01469) — *ICML*  
**One Step Diffusion via Shortcut Models** (2024) [[Paper]](https://arxiv.org/abs/2410.12557)  
**Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion** (2024) [[Paper]](https://arxiv.org/abs/2407.01392) — *NeurIPS*

### ⚡ Autoregressive & Transformer-Based

**VideoGPT: Video Generation using VQ-VAE and Transformers** (2021) [[Paper]](https://arxiv.org/abs/2104.10157)  
**GODIVA: Generating Open-Domain Videos from Natural Descriptions** (2021) [[Paper]](https://arxiv.org/abs/2104.14806)  
**CogVideo: Large-Scale Pretraining for Text-to-Video Generation via Transformers** (2022) [[Paper]](https://arxiv.org/abs/2205.15868)  
**Phenaki: Variable Length Video Generation from Open Domain Textual Description** (2022) [[Paper]](https://arxiv.org/abs/2210.02399)  
**Make-A-Video: Text-to-Video Generation without Text-Video Data** (2022) [[Paper]](https://arxiv.org/abs/2209.14792)  
**Temporally Consistent Transformers for Video Generation** (2023) [[Paper]](https://arxiv.org/abs/2210.02396) — *ICML*  
**iVideoGPT: Interactive VideoGPTs are Scalable World Models** (2024) [[Paper]](https://arxiv.org/abs/2405.15223) — *NeurIPS*  
**Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation** (2023) [[Paper]](https://arxiv.org/abs/2212.11565) — *ICCV*  
**Emu3: Next-Token Prediction is All You Need** (2024) [[Paper]](https://arxiv.org/abs/2409.18869)  
**Video Generation Models as World Simulators (Sora)** (2024) [[Paper]](https://openai.com/research/video-generation-models-as-world-simulators) — *OpenAI*  
**Movie Gen: A Cast of Media Foundation Models** (2024) [[Paper]](https://arxiv.org/abs/2410.13720) — *Meta*  
**Veo 2** (2024) [[Blog]](https://blog.google/innovation-and-ai/models-and-research/google-labs/video-image-generation-update-december-2024/) — *Google DeepMind*  
**Veo 3: A Text-to-Video Generation System with Audio** (2025) [[Paper]](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf) — *Google DeepMind*  
**Kling AI** (2024) [[Link]](https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-launches-25-turbo-video-model-industry-leading) — *Kuaishou*  
**Dream Machine** (2024) [[Link]](https://lumalabs.ai/dream-machine) — *Luma Labs*  
**Gen-3 Alpha** (2024) [[Link]](https://runwayml.com/research/introducing-gen-3-alpha) — *Runway*

### 🎞️ Long Video Generation

**SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2410.23277)  
**NUWA-XL: Diffusion over Diffusion for Extremely Long Video Generation** (2023) [[Paper]](https://arxiv.org/abs/2303.12346) — *ACL*  
**MALT Diffusion: Memory-Augmented Latent Transformers for Any-Length Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2502.12632)  
**Mixture of Contexts for Long Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2508.21058)  
**One-Minute Video Generation with Test-Time Training** (2025) [[Paper]](https://arxiv.org/abs/2504.05298) — *CVPR*  
**Long Context Tuning for Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2503.10589)  
**Packing Input Frame Context in Next-Frame Prediction Models for Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2504.12626)  
**Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models** (2025) [[Paper]](https://arxiv.org/abs/2412.05396) — *NeurIPS*  
**The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control** (2024) [[Paper]](https://arxiv.org/abs/2412.03568)  
**Error Analyses of Auto-Regressive Video Diffusion Models: A Unified Framework** (2025) [[Paper]](https://arxiv.org/abs/2503.10704)

### 🎛️ Controllable Video Generation

**DragNUWA: Fine-Grained Control in Video Generation by Integrating Text, Image, and Trajectory** (2023) [[Paper]](https://arxiv.org/abs/2308.08089)  
**MotionCtrl: A Unified and Flexible Motion Controller for Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2312.03641) — *SIGGRAPH*  
**Tora: Trajectory-Oriented Diffusion Transformer for Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2407.21705) — *CVPR*  
**DynamiCrafter: Animating Open-Domain Images with Video Diffusion Priors** (2024) [[Paper]](https://arxiv.org/abs/2310.12190) — *ECCV*  
**Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)** (2023) [[Paper]](https://arxiv.org/abs/2302.05543)  
**InstructVideo: Instructing Video Diffusion Models with Human Feedback** (2024) [[Paper]](https://arxiv.org/abs/2312.12490) — *CVPR*  
**InteractiveVideo: User-Centric Controllable Video Generation with Synergistic Multimodal Instructions** (2024) [[Paper]](https://arxiv.org/abs/2402.03040)  
**ATI: Any Trajectory Instruction for Controllable Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2505.22944)  
**AID: Adapting Image2Video Diffusion Models for Instruction-Guided Video Prediction** (2025) [[Paper]](https://arxiv.org/abs/2406.06465) — *ICCV*  
**Training-Free Guidance in Text-to-Video Generation via Multimodal Planning and Structured Noise Initialization** (2025) [[Paper]](https://arxiv.org/abs/2504.08641)  
**Force Prompting: Video Generation Models Can Learn and Generalize Physics-Based Control Signals** (2025) [[Paper]](https://arxiv.org/abs/2505.19386)  
**FreeAction: Training-Free Techniques for Enhanced Fidelity of Trajectory-to-Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2509.24241)  
**ViMi: Grounding Video Generation through Multi-Modal Instruction** (2024) [[Paper]](https://arxiv.org/abs/2311.01404) — *EMNLP*

---

## 🌍 World Models

### 🏛️ Foundational World Models

**World Models** (2018) [[Paper]](https://arxiv.org/abs/1803.10122) — Ha & Schmidhuber  
**Recurrent World Models Facilitate Policy Evolution** (2018) [[Paper]](https://arxiv.org/abs/1809.01999) — *NeurIPS*  
**Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)** (2020) [[Paper]](https://arxiv.org/abs/1912.01603) — *ICLR*  
**Mastering Atari with Discrete World Models (DreamerV2)** (2021) [[Paper]](https://arxiv.org/abs/2010.02193)  
**Mastering Diverse Domains through World Models (DreamerV3)** (2023) [[Paper]](https://arxiv.org/abs/2301.04104)  
**Training Agents Inside of Scalable World Models** (2025) [[Paper]](https://arxiv.org/abs/2509.24527)  
**Learning Interactive Real-World Simulators** (2023) [[Paper]](https://arxiv.org/abs/2310.06114)  
**Video as the New Language for Real-World Decision Making** (2024) [[Paper]](https://arxiv.org/abs/2402.17139)  
**Pandora: Towards General World Model with Natural Language Actions and Video States** (2024) [[Paper]](https://arxiv.org/abs/2406.09455)  
**AdaWorld: Learning Adaptable World Models with Latent Actions** (2025) [[Paper]](https://arxiv.org/abs/2503.18938)

### 🧠 Latent-Space World Models (for RL)

**TD-MPC2: Scalable, Robust World Models for Continuous Control** (2023) [[Paper]](https://arxiv.org/abs/2310.16828)  
**Diffusion World Model: Future Modeling Beyond Step-by-Step Rollout for Offline RL** (2024) [[Paper]](https://arxiv.org/abs/2402.03570)  
**DINO-WM: World Models on Pre-Trained Visual Features Enable Zero-Shot Planning** (2024) [[Paper]](https://arxiv.org/abs/2411.04983)  
**Robotic World Model: A Neural Network Simulator for Robust Policy Optimization** (2025) [[Paper]](https://arxiv.org/abs/2501.10100)  
**Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND)** (2024) [[Paper]](https://arxiv.org/abs/2405.12399) — *NeurIPS*  
**Pre-Training Contextualized World Models with In-the-Wild Videos for RL** (2023) [[Paper]](https://arxiv.org/abs/2305.18499) — *NeurIPS*  
**Pre-Trained Video Generative Models as World Simulators** (2025) [[Paper]](https://arxiv.org/abs/2502.07825)  
**Reinforcement Learning with Action-Free Pre-Training from Videos** (2022) [[Paper]](https://arxiv.org/abs/2203.13880) — *ICML*  
**DaydReamer: World Models for Physical Robot Learning** (2023) [[Paper]](https://arxiv.org/abs/2206.14176) — *CoRL*  
**Back to the Features: DINO as a Foundation for Video World Models** (2025) [[Paper]](https://arxiv.org/abs/2507.19468)  
**World4RL: Diffusion World Models for Policy Refinement with RL for Robotic Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2509.19080)

### 🔮 Video Diffusion as World Model

**Vid2World: Crafting Video Diffusion Models to Interactive World Models** (2025) [[Paper]](https://arxiv.org/abs/2505.14357)  
**AVID: Adapting Video Diffusion Models to World Models** (2024) [[Paper]](https://arxiv.org/abs/2410.12822)  
**Structured World Models from Human Videos** (2023) [[Paper]](https://arxiv.org/abs/2308.10901)  
**Ctrl-World: A Controllable Generative World Model for Robot Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2510.10125)  
**RoboDreamer: Learning Compositional World Models for Robot Imagination** (2024) [[Paper]](https://arxiv.org/abs/2404.12377)  
**Empowering World Models with Reflection for Embodied Video Prediction** (2025) [[Paper]](https://arxiv.org/abs/2410.15461) — *ICML*  
**EVA: An Embodied World Model for Future Video Anticipation** (2024) [[Paper]](https://arxiv.org/abs/2410.15461)  
**EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2501.01895)  
**EnerVerse-AC: Envisioning Embodied Environments with Action Condition** (2025) [[Paper]](https://arxiv.org/abs/2505.09723)  
**Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2508.05635)  
**World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training** (2025) [[Paper]](https://arxiv.org/abs/2509.24948)  
**1X World Model: Evaluating Bits, not Atoms** (2025) [[Paper]](https://www.1x.tech/discover/1x-world-model) — *1X Technologies*  
**Cosmos World Foundation Model Platform for Physical AI** (2025) [[Paper]](https://arxiv.org/abs/2501.03575) — *NVIDIA*  
**World Simulation With Video Foundation Models for Physical AI** (2025) [[Paper]](https://arxiv.org/abs/2511.00062) — *NVIDIA*  
**Aether: Geometric-Aware Unified World Modeling** (2025) [[Paper]](https://arxiv.org/abs/2503.18945)

### 🌐 3D/4D World Models

**Learning 3D Persistent Embodied World Models** (2025) [[Paper]](https://arxiv.org/abs/2505.05495)  
**Geometry-Aware 4D Video Generation for Robot Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2507.01099)  
**WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2510.07313)  
**WonderWorld: Interactive 3D Scene Generation from a Single Image** (2025) [[Paper]](https://arxiv.org/abs/2406.09394) — *CVPR*  
**WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions** (2025) [[Paper]](https://arxiv.org/abs/2505.18151)  
**4Diffusion: Multi-View Video Diffusion Model for 4D Generation** (2024) [[Paper]](https://arxiv.org/abs/2405.20674)  
**Splat4D: Diffusion-Enhanced 4D Gaussian Splatting** (2025) [[Paper]](https://arxiv.org/abs/2508.07557)  
**3D Gaussian Splatting for Real-Time Radiance Field Rendering** (2023) [[Paper]](https://arxiv.org/abs/2308.04079)  
**MindJourney: Test-Time Scaling with World Models for Spatial Reasoning** (2025) [[Paper]](https://arxiv.org/abs/2507.12508)

### 🎮 Interactive & Game World Models

**Genie: Generative Interactive Environments** (2024) [[Paper]](https://arxiv.org/abs/2402.15391) — *ICML*  
**Genie 2: A Large-Scale Foundation World Model** (2024) [[Blog]](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) — *Google DeepMind*  
**Genie 3: A New Frontier for World Models** (2025) [[Blog]](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) — *Google DeepMind*  
**GameGen-X: Interactive Open-World Game Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2411.00769)  
**Navigation World Models** (2025) [[Paper]](https://arxiv.org/abs/2412.03572) — *CVPR*  
**WoMAP: World Models for Embodied Open-Vocabulary Object Localization** (2025) [[Paper]](https://arxiv.org/abs/2506.01600) — *RSS Workshop*  
**PlaySlot: Learning Inverse Latent Dynamics for Controllable Object-Centric Video Prediction** (2025) [[Paper]](https://arxiv.org/abs/2502.07600)  
**WorldGym: World Model as An Environment for Policy Evaluation** (2025) [[Paper]](https://arxiv.org/abs/2506.00613)

---

## 🤖 Robotics Applications

### 🎓 Robot Policy Learning from Video

**Visual Foresight: Model-Based Deep RL for Vision-Based Robotic Control** (2018) [[Paper]](https://arxiv.org/abs/1812.00568)  
**Deep Visual Foresight for Planning Robot Motion** (2017) [[Paper]](https://arxiv.org/abs/1610.00696)  
**Learning What You Can Do Before Doing Anything** (2018) [[Paper]](https://arxiv.org/abs/1806.09655)  
**Zero-Shot Visual Imitation** (2018) [[Paper]](https://arxiv.org/abs/1804.08606) — *CVPR*  
**Learning to Poke by Poking: Experiential Learning of Intuitive Physics** (2016) [[Paper]](https://arxiv.org/abs/1606.07419) — *NeurIPS*  
**Video Pretraining (VPT): Learning to Act by Watching Unlabeled Online Videos** (2022) [[Paper]](https://arxiv.org/abs/2206.11795) — *NeurIPS*  
**Latent Action Pretraining from Videos (LAPA)** (2024) [[Paper]](https://arxiv.org/abs/2410.11758) — *ICLR*  
**Learning to Act without Actions** (2024) [[Paper]](https://arxiv.org/abs/2312.10812) — *ICLR*  
**Robot Learning with Sensorimotor Pre-Training** (2023) [[Paper]](https://arxiv.org/abs/2306.10007) — *CoRL*  
**Unleashing Large-Scale Video Generative Pre-Training for Visual Robot Manipulation** (2023) [[Paper]](https://arxiv.org/abs/2312.13139)  
**Gen2Act: Human Video Generation in Novel Scenarios Enables Generalizable Robot Manipulation** (2024) [[Paper]](https://arxiv.org/abs/2409.16283)  
**DreamGen: Unlocking Generalization in Robot Learning through Video World Models** (2025) [[Paper]](https://arxiv.org/abs/2505.12705)  
**LuciBot: Automated Robot Policy Learning from Generated Videos** (2025) [[Paper]](https://arxiv.org/abs/2503.09871)  
**Human2Robot: Learning Robot Actions from Paired Human-Robot Videos** (2025) [[Paper]](https://arxiv.org/abs/2502.16587)  
**Solving New Tasks by Adapting Internet Video Knowledge** (2025) [[Paper]](https://arxiv.org/abs/2504.15369)  
**mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs** (2025) [[Paper]](https://arxiv.org/abs/2512.15692)  
**Video Generators are Robot Policies** (2025) [[Paper]](https://arxiv.org/abs/2508.00795)

### 📋 Video as Robot Action Planner

**Learning Universal Policies via Text-Guided Video Generation (UniPi)** (2023) [[Paper]](https://arxiv.org/abs/2302.00111) — *NeurIPS*  
**Video Language Planning** (2023) [[Paper]](https://arxiv.org/abs/2310.10625)  
**Compositional Foundation Models for Hierarchical Planning** (2023) [[Paper]](https://arxiv.org/abs/2309.08587) — *NeurIPS*  
**Grounding Video Models to Actions through Goal Conditioned Exploration** (2024) [[Paper]](https://arxiv.org/abs/2411.07223)  
**Learning to Act from Actionless Videos through Dense Correspondences** (2023) [[Paper]](https://arxiv.org/abs/2310.08576)  
**Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models (SuSIE)** (2023) [[Paper]](https://arxiv.org/abs/2310.10639)  
**Dreamitate: Real-World Visuomotor Policy Learning via Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2406.16862)  
**Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations** (2025) [[Paper]](https://arxiv.org/abs/2507.00990)  
**NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos** (2025) [[Paper]](https://arxiv.org/abs/2510.08568)  
**VILP: Imitation Learning with Latent Video Planning** (2025) [[Paper]](https://arxiv.org/abs/2406.01897) — *RA-L*  
**Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations** (2024) [[Paper]](https://arxiv.org/abs/2412.14803)  
**VidMan: Exploiting Implicit Dynamics from Video Diffusion Model for Effective Robot Manipulation** (2024) [[Paper]](https://arxiv.org/abs/2411.09153) — *NeurIPS*  
**Prediction with Action: Visual Policy Learning via Joint Denoising Process** (2024) [[Paper]](https://arxiv.org/abs/2402.02735) — *NeurIPS*  
**GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge** (2024) [[Paper]](https://arxiv.org/abs/2410.06158)  
**Unified Video Action Model** (2025) [[Paper]](https://arxiv.org/abs/2503.00200)  
**Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets** (2025) [[Paper]](https://arxiv.org/abs/2504.02792)  
**FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model** (2024) [[Paper]](https://arxiv.org/abs/2412.08261)  
**LaDi-WM: A Latent Diffusion-Based World Model for Predictive Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2505.11528)  
**This&That: Language-Gesture Controlled Video Generation for Robot Planning** (2025) [[Paper]](https://arxiv.org/abs/2407.05530) — *ICRA*  
**ARDuP: Active Region Video Diffusion for Universal Policies** (2024) [[Paper]](https://arxiv.org/abs/2406.13301) — *IROS*  
**Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation** (2024) [[Paper]](https://arxiv.org/abs/2409.09016) — *NeurIPS*  
**Large Video Planner Enables Generalizable Robot Control** (2025) [[Paper]](https://arxiv.org/abs/2512.15840)  
**Implicit State Estimation via Video Replanning** (2025) [[Paper]](https://arxiv.org/abs/2510.17315)  
**VideoAgent: Self-Improving Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2410.10076)  
**IRASimt: Learning Interactive Real-Robot Action Simulators** (2024) [[Paper]](https://arxiv.org/abs/2406.14540)  
**Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control** (2025) [[Paper]](https://arxiv.org/abs/2506.01943)  
**RoboEnvision: A Long-Horizon Video Generation Model for Multi-Task Robot Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2506.22007)  
**ViDaR: Embodied Video Diffusion Model for Generalist Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2507.12898)  
**AnyPos: Automated Task-Agnostic Actions for Bimanual Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2507.12768)  
**TASTE-Rob: Advancing Video Generation of Task-Oriented Hand-Object Interaction** (2025) [[Paper]](https://arxiv.org/abs/2503.10945) — *CVPR*  
**GEVRm: Goal-Expressive Video Generation Model for Robust Visual Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2502.09268)  
**DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge** (2025) [[Paper]](https://arxiv.org/abs/2507.04447)  
**VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation** (2024) [[Paper]](https://arxiv.org/abs/2407.09829)

### 🏆 Reward Learning from Video

**Video Prediction Models as Rewards for Reinforcement Learning (VIPER)** (2023) [[Paper]](https://arxiv.org/abs/2305.14343) — *NeurIPS*  
**Diffusion Reward: Learning Rewards via Conditional Video Diffusion** (2024) [[Paper]](https://arxiv.org/abs/2312.14134) — *ECCV*  
**LIV: Language-Image Representations and Rewards for Robotic Control** (2023) [[Paper]](https://arxiv.org/abs/2306.00958) — *ICML*

### ✅ Policy Evaluation with World Models

**Evaluating Real-World Robot Manipulation Policies in Simulation (SIMPLER)** (2024) [[Paper]](https://arxiv.org/abs/2405.05941)  
**Scalable Policy Evaluation with Video World Models** (2025) [[Paper]](https://arxiv.org/abs/2511.11520)  
**WorldEval: World Model as Real-World Robot Policies Evaluator** (2025) [[Paper]](https://arxiv.org/abs/2505.19017)  
**Evaluating Gemini Robotics Policies in a Veo World Simulator** (2025) [[Paper]](https://arxiv.org/abs/2512.10675) — *Google DeepMind*  
**PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies** (2025) [[Paper]](https://arxiv.org/abs/2512.16881)  
**Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions** (2025) [[Paper]](https://arxiv.org/abs/2511.04665)  
**DiWa: Diffusion Policy Adaptation with World Models** (2025) [[Paper]](https://arxiv.org/abs/2508.03645)  
**RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies** (2025) [[Paper]](https://arxiv.org/abs/2506.18123)  
**RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots** (2024) [[Paper]](https://arxiv.org/abs/2406.02523)

### 💬 Vision-Language-Action Models

**RT-1: Robotics Transformer for Real-World Control at Scale** (2023) [[Paper]](https://arxiv.org/abs/2212.06817)  
**RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control** (2023) [[Paper]](https://arxiv.org/abs/2307.15818) — *CoRL*  
**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (2023) [[Paper]](https://arxiv.org/abs/2303.04137)  
**ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware** (2023) [[Paper]](https://arxiv.org/abs/2304.13705)  
**π₀.5: A Vision-Language-Action Model with Open-World Generalization** (2025) [[Paper]](https://arxiv.org/abs/2504.16054)  
**Gemini Robotics: Bringing AI into the Physical World** (2025) [[Paper]](https://arxiv.org/abs/2503.20020)  
**MolmoAct: Action Reasoning Models that Can Reason in Space** (2025) [[Paper]](https://arxiv.org/abs/2508.07917)  
**Actions as Language: Fine-Tuning VLMs into VLAs without Catastrophic Forgetting** (2025) [[Paper]](https://arxiv.org/abs/2509.22195)  
**Unified Vision-Language-Action Model** (2025) [[Paper]](https://arxiv.org/abs/2506.19850)  
**ViLLA-X: Enhancing Latent Action Modeling in Vision-Language-Action Models** (2025) [[Paper]](https://arxiv.org/abs/2507.23682)  
**ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation** (2024) [[Paper]](https://arxiv.org/abs/2409.01652)  
**Using Left and Right Brains Together: Towards Vision and Language Planning** (2024) [[Paper]](https://arxiv.org/abs/2402.10534)  
**WoMAP: World Models for Embodied Open-Vocabulary Object Localization** (2025) [[Paper]](https://arxiv.org/abs/2506.01600) — *RSS Workshop*

---

## 🚗 Autonomous Driving World Models

**GAIA-1: A Generative World Model for Autonomous Driving** (2023) [[Paper]](https://arxiv.org/abs/2309.17080)  
**GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving** (2025) [[Paper]](https://arxiv.org/abs/2503.20523)  
**Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability** (2024) [[Paper]](https://arxiv.org/abs/2405.17398) — *NeurIPS*  
**VaViM and VaVAM: Autonomous Driving through Video Generative Modeling** (2025) [[Paper]](https://arxiv.org/abs/2502.15672)  
**DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving** (2025) [[Paper]](https://arxiv.org/abs/2510.12796)  
**SCOPE: Stochastic Cartographic Occupancy Prediction Engine for Uncertainty-Aware Dynamic Navigation** (2025) [[Paper]](https://ieeexplore.ieee.org/document/11029153/) — *IEEE Transactions on Robotics*

---

## 🔍 Self-Supervised Video Representations

**CLIP: Learning Transferable Visual Models from Natural Language Supervision** (2021) [[Paper]](https://arxiv.org/abs/2103.00020) — *ICML*  
**ViViT: A Video Vision Transformer** (2021) [[Paper]](https://arxiv.org/abs/2103.15691) — *ICCV*  
**DINO: Emerging Properties in Self-Supervised Vision Transformers** (2021) [[Paper]](https://arxiv.org/abs/2104.14294) — *ICCV*  
**DINOv2: Learning Robust Visual Features without Supervision** (2024) [[Paper]](https://arxiv.org/abs/2304.07193)  
**SimCLR: A Simple Framework for Contrastive Learning of Visual Representations** (2020) [[Paper]](https://arxiv.org/abs/2002.05709)  
**BYOL: Bootstrap Your Own Latent** (2020) [[Paper]](https://arxiv.org/abs/2006.07733)  
**VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning** (2022) [[Paper]](https://arxiv.org/abs/2105.04906)  
**I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture** (2023) [[Paper]](https://arxiv.org/abs/2301.08243) — *CVPR*  
**V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video** (2024) [[Paper]](https://arxiv.org/abs/2404.08471)  
**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** (2025) [[Paper]](https://arxiv.org/abs/2506.09985)  
**LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics** (2025) [[Paper]](https://arxiv.org/abs/2511.08544)  
**Intuitive Physics Understanding Emerges from Self-Supervised Pretraining on Natural Videos** (2025) [[Paper]](https://arxiv.org/abs/2502.11831)  
**Video Models are Zero-Shot Learners and Reasoners** (2025) [[Paper]](https://arxiv.org/abs/2509.20328)

---

## 🤲 Affordances from Video

**Demo2Vec: Reasoning Object Affordances from Online Videos** (2018) [[Paper]](https://arxiv.org/abs/1812.00371) — *CVPR*  
**Grounded Human-Object Interaction Hotspots from Video** (2019) [[Paper]](https://arxiv.org/abs/1906.10279) — *ICCV*  
**Forecasting Human-Object Interaction: Joint Prediction of Motor Attention and Actions** (2020) [[Paper]](https://arxiv.org/abs/1911.10967) — *ECCV*  
**Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos** (2022) [[Paper]](https://arxiv.org/abs/2204.01696) — *CVPR*  
**Discovering a Variety of Objects in Spatio-Temporal Human-Object Interactions** (2022) [[Paper]](https://arxiv.org/abs/2211.07501)  
**Fine-Grained Affordance Annotation for Egocentric Hand-Object Interaction Videos** (2023) [[Paper]](https://arxiv.org/abs/2302.01649) — *WACV*  
**Multi-Label Affordance Mapping from Egocentric Vision** (2023) [[Paper]](https://arxiv.org/abs/2309.09269) — *ICCV*  
**VRB: Affordances from Human Videos as a Versatile Representation for Robotics** (2023) [[Paper]](https://arxiv.org/abs/2304.08488) — *CVPR*  
**Robo-ABC: Affordance Generalization Beyond Categories via Semantic Correspondence** (2024) [[Paper]](https://arxiv.org/abs/2401.07487) — *ECCV*  
**Learning Precise Affordances from Egocentric Videos for Robotic Manipulation** (2025) [[Paper]](https://arxiv.org/abs/2408.10123) — *ICCV*

---

## ⚗️ Physical Understanding & Benchmarks

### ⚛️ Physics in Video Models

**How Far is Video Generation from World Model: A Physical Law Perspective** (2024) [[Paper]](https://arxiv.org/abs/2411.02385)  
**Do Generative Video Models Understand Physical Principles?** (2025) [[Paper]](https://arxiv.org/abs/2501.09038)  
**VideoPhY: Evaluating Physical Commonsense for Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2406.03520)  
**Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2410.05363)  
**WISA: World Simulator Assistant for Physics-Aware Text-to-Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2503.08153)  
**Think Before You Diffuse: LLMs-Guided Physics-Aware Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2505.21653)  
**PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2409.18964) — *ECCV*  
**Hamiltonian Neural Networks** (2019) [[Paper]](https://arxiv.org/abs/1906.01563)  
**LagNetViP: A Lagrangian Neural Network for Video Prediction** (2020) [[Paper]](https://arxiv.org/abs/2010.12932)  
**Physics-Informed Machine Learning** (2021) [[Paper]](https://www.nature.com/articles/s42254-021-00314-5) — *Nature Reviews Physics*  
**PhysMaster: Mastering Physical Representation for Video Generation via Reinforcement Learning** (2025) [[Paper]](https://arxiv.org/abs/2510.13809)  
**Hierarchical Fine-Grained Preference Optimization for Physically Plausible Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2508.10858)  
**MoAlign: Motion-Centric Representation Alignment for Video Diffusion Models** (2025) [[Paper]](https://arxiv.org/abs/2510.19022)  
**A Shortcut-Aware Video-QA Benchmark for Physical Understanding via Minimal Video Pairs** (2025) [[Paper]](https://arxiv.org/abs/2506.09987)

### 📏 Video Generation Benchmarks

**VBench: Comprehensive Benchmark Suite for Video Generative Models** (2024) [[Paper]](https://arxiv.org/abs/2311.17982) — *CVPR*  
**EvalCrafter: Benchmarking and Evaluating Large Video Generation Models** (2024) [[Paper]](https://arxiv.org/abs/2310.11440) — *CVPR*  
**T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2407.14505) — *CVPR*  
**WorldSimBench: Towards Video Generation Models as World Simulators** (2024) [[Paper]](https://arxiv.org/abs/2410.18072)  
**WorldModelBench: Judging Video Generation Models as World Models** (2025) [[Paper]](https://arxiv.org/abs/2502.20694)  
**EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models** (2025) [[Paper]](https://arxiv.org/abs/2505.09694)  
**A Control-Centric Benchmark for Video Prediction** (2023) [[Paper]](https://arxiv.org/abs/2304.13723)  
**DynamicEval: Rethinking Evaluation for Dynamic Text-to-Video Synthesis** (2025) [[Paper]](https://arxiv.org/abs/2510.07441)  
**Physical AI Bench** (2025) [[Link]](https://github.com/SHI-Labs/physical-ai-bench)  
**VIBE: A Text-to-Video Benchmark for Evaluating Hallucination in Large Multimodal Models** (2025) [[Paper]](https://arxiv.org/abs/2406.09400) — *TrustNLP*  
**Sora Detector: A Unified Hallucination Detection for Large Text-to-Video Models** (2024) [[Paper]](https://arxiv.org/abs/2405.04180)  
**Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)** (2004) [[Paper]](https://ieeexplore.ieee.org/document/1284395) — *IEEE TIP*  
**The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)** (2018) [[Paper]](https://arxiv.org/abs/1801.03924) — *CVPR*  
**Towards Accurate Generative Models of Video: A New Metric & Challenges (FVD)** (2018) [[Paper]](https://arxiv.org/abs/1812.01717)  
**GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)** (2017) [[Paper]](https://arxiv.org/abs/1706.08500)  
**DreamSim: Learning New Dimensions of Human Visual Similarity** (2023) [[Paper]](https://arxiv.org/abs/2306.09344)  
**TempCompass: Do Video LLMs Really Understand Videos?** (2024) [[Paper]](https://arxiv.org/abs/2403.00476)  
**TemporalBench: Benchmarking Fine-Grained Temporal Understanding for Multimodal Video Models** (2024) [[Paper]](https://arxiv.org/abs/2410.10818)  
**TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models** (2025) [[Paper]](https://arxiv.org/abs/2410.23266)  
**MVBench: A Comprehensive Multi-Modal Video Understanding Benchmark** (2024) [[Paper]](https://arxiv.org/abs/2311.17005)  
**TVBench: Redesigning Video-Language Evaluation** (2025) [[Paper]](https://openreview.net/forum?id=DrNN5qx66Z)  
**Perception Test: A Diagnostic Benchmark for Multimodal Video Models** (2023) [[Paper]](https://arxiv.org/abs/2305.13786)

### 💾 Robotic Datasets

**Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets** (2021) [[Paper]](https://arxiv.org/abs/2109.13396)  
**BridgeData V2: A Dataset for Robot Learning at Scale** (2023) [[Paper]](https://arxiv.org/abs/2308.12952) — *CoRL*  
**Open X-Embodiment: Robotic Learning Datasets and RT-X Models** (2024) [[Paper]](https://arxiv.org/abs/2310.08864) — *ICRA*  
**DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset** (2024) [[Paper]](https://arxiv.org/abs/2403.12945)  
**LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning** (2023) [[Paper]](https://arxiv.org/abs/2306.03310)  
**Something-Something Video Database** (2017) [[Paper]](https://arxiv.org/abs/1706.04261)  
**Kinetics: The Kinetics Human Action Video Dataset** (2017) [[Paper]](https://arxiv.org/abs/1705.06950)  
**EPIC-KITCHENS-100** (2020) [[Dataset]](https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m/)  
**Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers** (2024) [[Paper]](https://arxiv.org/abs/2402.19479) — *CVPR*  
**VidGen-1M: A Large-Scale Dataset for Text-to-Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2408.02629)  
**OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-Video Generation** (2024) [[Paper]](https://arxiv.org/abs/2407.02371)  
**Koala-36M: A Large-Scale Video Dataset** (2025) [[Paper]](https://arxiv.org/abs/2501.05905) — *CVPR*

---

## ❓ Uncertainty in World Models

**How Confident are Video Models? Empowering Video Models to Express their Uncertainty** (2025) [[Paper]](https://arxiv.org/abs/2510.02571)  
**World Models That Know When They Don't Know: Controllable Video Generation with Calibrated Uncertainty** (2025) [[Paper]](https://arxiv.org/abs/2512.05927)  
**Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know?** (2025) [[Paper]](https://arxiv.org/abs/2506.18183)  
**Shedding Light on Large Generative Networks: Estimating Epistemic Uncertainty in Diffusion Models** (2024) [[Paper]](https://arxiv.org/abs/2406.01710) — *UAI*  
**Estimating Epistemic and Aleatoric Uncertainty with a Single Model** (2024) [[Paper]](https://arxiv.org/abs/2402.03289) — *NeurIPS*  
**Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation** (2025) [[Paper]](https://arxiv.org/abs/2412.06024) — *CVPR*  
**A Survey on Uncertainty Quantification of Large Language Models** (2025) [[Paper]](https://arxiv.org/abs/2412.05563)

---

## 🛡️ Safety

**The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems** (2024) [[Paper]](https://www.annualreviews.org/content/journals/10.1146/annurev-control-071723-102940) — *Annual Review of Control, Robotics, and Autonomous Systems*  
**Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis** (2025) [[Paper]](https://arxiv.org/abs/2502.00935)  
**Uncertainty-Aware Latent Safety Filters for Avoiding Out-of-Distribution Failures** (2025) [[Paper]](https://arxiv.org/abs/2505.00779)  
**AnySafe: Adapting Latent Safety Filters at Runtime** (2025) [[Paper]](https://arxiv.org/abs/2509.19555)  
**SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation** (2025) [[Paper]](https://arxiv.org/abs/2410.12761)  
**SafeWatch: An Efficient Safety-Policy Following Video Guardrail Model** (2024) [[Paper]](https://arxiv.org/abs/2412.06878)  
**T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models** (2024) [[Paper]](https://arxiv.org/abs/2407.05965)  
**Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs** (2025) [[Paper]](https://arxiv.org/abs/2505.11842)  
**SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents** (2025) [[Paper]](https://arxiv.org/abs/2412.13178)  
**Safeguarding Large Language Models: A Survey** (2024) [[Paper]](https://arxiv.org/abs/2406.02622)
