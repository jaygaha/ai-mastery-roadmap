# Exercises or Practice Activities

To reinforce your understanding of the foundational concepts of AI and Machine Learning, try the following activities:

## Activity 1: Identify the Learning Paradigm

For each scenario below, identify which machine learning paradigm (Supervised Learning, Unsupervised Learning, or Reinforcement Learning) would be most appropriate and briefly explain why.

- **Scenario A**: A streaming service wants to recommend new movies to users based on their past viewing history and ratings. For each movie a user has watched, they also provided a rating (1-5 stars).

  - Paradigm: ??
    > Supervised Learning
  - Explanation: ??
    > This scenario involves labeled data (movie ratings as explicit feedback from users) and the goal is to predict future ratings or preferences based on historical inputs. Supervised learning algorithms, such as regression or classification models, can be trained on this data to recommend movies likely to receive high ratings from similar users.
- **Scenario B**: A pharmaceutical company wants to discover new subgroups of patients who respond similarly to a particular drug, even though they haven't explicitly defined these groups before. They have a lot of patient data including genetics, medical history, and drug dosage, but no pre-defined patient type labels.

  - Paradigm: 
    > Unsupervised Learning
  - Explanation:
    > This scenario involves unlabeled data where the goal is to identify hidden patterns or clusters within the patient data without predefined categories. Unsupervised learning techniques, such as clustering algorithms (e.g., K-means), can group patients based on similarities in their features like genetics and medical history, revealing subgroups that respond similarly to the drug.
- **Scenario C**: An AI is being developed to manage the temperature settings in a smart home to minimize energy consumption while keeping residents comfortable. The AI adjusts the thermostat and learns over time based on feedback (e.g., "too hot," "too cold," or energy bill reduction).

  - Paradigm: ??
    > Reinforcement Learning
  - Explanation: ??
    > The AI learns by interacting with the environment (adjusting the thermostat) and receiving feedback ("too hot," "too cold," or energy bill changes) to optimize its actions. Reinforcement learning is ideal here because the AI must balance competing goals (energy savings vs. comfort) through trial and error, learning a policy that maximizes a reward signal based on resident feedback and energy efficiency.
- **Scenario D**: A bank wants to automatically approve or deny credit card applications. They have historical data of thousands of applicants, including their income, credit score, debt-to-income ratio, and whether their previous applications were approved or denied.

  - Paradigm: ??
    > Supervised Learning
  - Explanation: ??
    > The bank has historical data with clear labels (approved or denied applications) paired with features like income, credit score, and debt-to-income ratio. Supervised learning is appropriate because the model can be trained on this labeled dataset to predict whether new applications should be approved or denied, treating it as a binary classification problem.

  
## Activity 2: Data, Features, and Labels/Outcomes

For each problem below, describe:

- What kind of data would be collected.
- What potential features would be relevant.
- If applicable, what the label or outcome would be.

- **Problem A: Predicting the likelihood of a customer clicking on an online advertisement.**:
  - Data: ??
    > Historical user interaction data from websites or apps, including ad impressions, user browsing history, demographic information, and click events.
  - Features: ??
    > Ad type (e.g., banner, video), user demographics (age, location), time of day, device type, previous click history, and ad content keywords.
  - Label/Outcome: ??
    > Binary label indicating whether the user clicked on the ad (e.g., 1 for click, 0 for no click), used to train a classification model.
- **Problem B: Grouping different types of music based on their audio characteristics (e.g., tempo, instrumentation, rhythm) without knowing the genres beforehand.**:
  - Data: ??
    > Raw audio files or spectrogram's of music tracks, possibly including metadata like duration but no genre labels.
  - Features: ??
    > Tempo (beats per minute), rhythm patterns, pitch, timbre, instrumentation (e.g., presence of guitar, drums), spectral features (e.g., frequency distribution), and energy levels.
  - Label/Outcome: ??
    > No predefined labels, as this is an unsupervised learning task. The outcome is clusters of music tracks grouped by similar audio characteristics.
- **Problem C: Training a robotic arm to pick up irregularly shaped objects from a conveyor belt without human intervention.**:
  - Data: ??
    > Sensor data from the robotic arm and conveyor belt, including camera images, depth sensor readings, and arm movement logs, along with feedback on task success or failure.
  - Features: ??
    > Object shape (from image/depth data), object size, weight (if measurable), position on the conveyor belt, surface texture, arm joint angles, and grip force.
  - Label/Outcome: ??
    > A reward signal in reinforcement learning, where a positive reward is given for successfully picking and placing an object (e.g., +1), and a penalty for failures like dropping or missing an object (e.g., -1).
     
## Activity 3: AI Type Classification

**Classify the following scenarios as examples of Narrow AI, General AI, or Superintelligence. Justify your answer.**

- **Scenario A**: An AI system that can beat the world's best human chess players.:
  - Classification: ??
    > Narrow AI
  - Justification:: ??
    > This AI is specialized for a single task (playing chess) and excels in that domain through pattern recognition and strategic computation, but it lacks general intelligence to perform unrelated tasks like creative writing or everyday problem-solving.
- **Scenario B**: A hypothetical AI capable of writing a bestselling novel, conducting complex scientific research in multiple fields, and developing new ethical philosophies that surpass human understanding.:
  - Classification: ??
    > Superintelligence
  - Justification:: ??
    >This AI surpasses human capabilities across diverse domains (creative writing, scientific research, and philosophical reasoning) and produces outcomes that exceed human understanding. Superintelligence is defined by its ability to outperform humans in virtually all intellectual tasks, making this a fitting classification.
- **Scenario C**: An AI algorithm used in a factory to detect defects in manufactured products by analyzing images from a production line camera.:
  - Classification: ??
    > Narrow AI
  - Justification:: ??
    > The AI is tailored for a single, specific task—defect detection in manufacturing—using image analysis. It operates within a constrained domain and does not exhibit general problem-solving abilities or capabilities beyond its designated function, aligning with Narrow AI.