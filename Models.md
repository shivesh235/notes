## Hands On LLM

### Language Models

The sources explain that **Language Models (LMs)** are computer systems designed to perform tasks related to human intelligence, such as speech recognition and language translation, embodying the intelligence of software. The history of **Language AI** includes many developments and models aiming to represent and generate language. This encompasses models from simple bag-of-words representations to more sophisticated dense vector embeddings like word2vec. These early models, like word2vec, aimed to capture the meaning of text in embeddings.

The sources then introduce **Large Language Models (LLMs)** as a more recent and impactful subset of language models. **LLMs** have enabled machines to better understand and generate human-like language, opening new possibilities in AI. The book emphasizes that **LLMs** are not just a single type of model, and the term can encompass both **generative models** (decoder-only, like GPT) that generate text, and **representation models** (encoder-only, like BERT) that focus on understanding and creating embeddings for tasks like classification. Although **generative models** might be the first thing that comes to mind when thinking about **LLMs**, **representation models** are still highly useful.

The term "**large**" in **large language models** is described as somewhat arbitrary and evolving with the release of new models. What is considered large today might be small tomorrow. The book takes a broad view, considering "**large language models**" to include models that do not necessarily generate text and can even be run on consumer hardware. Therefore, the book covers not only generative models but also smaller models (under 1 billion parameters) that are representation models.

The Transformer architecture is identified as a foundational element for many impactful models in **Language AI**, including both BERT and GPT-1, and most models used throughout the book are Transformer-based. These architectures, along with their training procedures, have led to significant advancements in the capabilities of **language models**.

In essence, the sources present "**Language Models**" as the overarching field of AI focused on understanding and processing language, with a history of various techniques for representing and generating text. "**Large Language Models**" are a more recent generation of **language models**, often characterized by their scale (though the definition of "large" is fluid) and capabilities, frequently built on the Transformer architecture. **LLMs** can be either generative or representational, and the book aims to provide a comprehensive understanding of both within the broader context of **Language AI**.

---

### Generative models (decoder-only)

These sources discuss **generative models**, specifically those with a **decoder-only architecture** (like the GPT family), as a significant category within the broader landscape of **Language Models (LMs)** and particularly **Large Language Models (LLMs)**.

Here's a breakdown of what the sources say about generative models in this larger context:

*   **Definition and Core Functionality:** Generative models are a type of LLM that **generate text**. They function as **sequence-to-sequence machines**, taking in some text and attempting to **autocomplete it**. They are often referred to as **completion models**. This is achieved through an architecture that **stacks decoder blocks**, similar to the encoder stacking in models like BERT, but **without the encoder-attention block** of the original Transformer.

*   **Relationship to LLMs:** Although the term "**LLM**" can encompass both generative (decoder-only) and representation (encoder-only) models, **generative models are typically the first thing that comes to mind when thinking about LLMs**. The sources highlight that the definition of "large" in LLMs is somewhat arbitrary and evolving.

*   **Examples and Evolution:** The **Generative Pre-trained Transformer (GPT) family**, starting with **GPT-1**, is the prime example of decoder-only generative models. These models, especially the larger iterations, have demonstrated remarkable capabilities in generating human-like text. The book mentions the rapid progress in this area, highlighting the "**Year of Generative AI**".

*   **Distinction from Representation Models:** The sources make a clear distinction between **representation models (encoder-only)** like BERT and **generative models (decoder-only)**. Representation models primarily focus on **representing language** by creating embeddings and are commonly used for task-specific use cases like classification. In contrast, **generative models primarily focus on generating text** and are typically not trained to generate embeddings. This distinction is even visually represented in the book's illustrations.

*   **Fine-tuning for Specific Tasks:** While base generative models are trained on vast amounts of text to predict the next token, they can be **fine-tuned** to perform specific tasks such as answering questions and following instructions, leading to the creation of "**instruct**" or "**chat**" models. This involves training the model on data that aligns with the desired behavior.

*   **Importance of Context:** A vital aspect of generative models is their **context length** or **context window**, which defines the maximum number of tokens they can process. A larger context window allows them to consider more information when generating text.

*   **Evaluation:** Generative models are evaluated using various methods, including **benchmarks** like **MMLU, GLUE, TruthfulQA, and GSM8k**. These benchmarks assess their language understanding and generation abilities. Additionally, the quality of their generated text can be evaluated using **automated metrics** and by employing other **LLMs as judges**.

*   **Prompt Engineering:** Interacting effectively with generative models often requires **prompt engineering**, which involves iteratively improving the prompts given to the model to guide it towards the desired output. Techniques like in-context learning and chain-of-thought prompting are used to enhance their reasoning abilities.

*   **Applications:** Generative models have a wide range of applications, including **text generation**, building **chatbots**, and even performing **text classification** by prompting them to generate a class label. They are also a key component in more complex systems like **Retrieval-Augmented Generation (RAG)**, where they use retrieved information to generate more factual answers.

In summary, the sources position **decoder-only generative models** as a central and rapidly evolving area within the field of Language Models. They are distinguished by their text generation capabilities and are often what people associate with the power of LLMs. While different from encoder-only representation models, both types contribute significantly to the broader landscape of Language AI, and understanding their distinctions and functionalities is crucial.

* **GPT family of models (GPT-1, GPT-2, GPT-3.5, GPT-4, and Phi-3)**
* **Llama family (Llama 2)**


#### GPT family of models (GPT-1, GPT-2, GPT-3.5, GPT-4, and Phi-3)

These sources place the **GPT family of models (GPT-1, GPT-2, GPT-3.5, GPT-4, and Phi-3)** firmly within the category of **decoder-only generative models**, which are a significant part of the larger landscape of **Language Models (LMs)** and **Large Language Models (LLMs)**.

Here's a breakdown of what the sources say about them in this context:

*   **Decoder-Only Architecture and Generative Task:** The sources explicitly state that the GPT family employs a **decoder-only architecture**, contrasting it with the encoder-only architecture of models like BERT. This architecture is specifically designed for **generative tasks**, where the model takes in text and attempts to **autocomplete it**, functioning as a **sequence-to-sequence machine**. These models are also referred to as **completion models**. Figure 1-24 illustrates the architecture of GPT-1, highlighting the stacking of decoder blocks and the removal of the encoder-attention block.

*   **GPT-1 as the Foundation:** The first model in this family, **GPT-1 (Generative Pre-trained Transformer)**, was introduced in 2018 and marked a significant step in targeting generative tasks. It was trained on a large corpus of books and web pages (Common Crawl) and consisted of **117 million parameters**. GPT-1 established the decoder-only Transformer as a foundation for future generative models.

*   **GPT-2 and Scaling:** **GPT-2**, released later, demonstrated the impact of **scaling up the number of parameters**, increasing them to **1.5 billion**. The tokenizer used by GPT-2 utilizes **Byte Pair Encoding (BPE)** and includes the special token `<|endoftext|>`. The source notes that GPT-2's tokenizer preserves capitalization and represents newline breaks.

*   **GPT-3.5 and ChatGPT:** The release and adoption of **ChatGPT**, powered initially by the **GPT-3.5 LLM**, is highlighted as a pivotal moment, leading some to call 2023 "**The Year of Generative AI**". While the underlying architecture of the original ChatGPT (GPT-3.5) is not shared, it is assumed to be based on the decoder-only architecture of the GPT models. The training of GPT-3.5 involved **preference tuning** based on manually created instruction data and ranked model outputs, which allowed it to better align with human preferences.

*   **GPT-4's Capabilities:** **GPT-4** is presented as a more performant variant of the GPT family. Its tokenizer also uses **BPE** but with a significantly larger vocabulary size of over **100,000**. The GPT-4 tokenizer has specific tokens for sequences of whitespaces and Python keywords, reflecting its focus on code in addition to natural language. It also generally uses fewer tokens to represent most words compared to GPT-2. GPT-4 also incorporates special "**fill in the middle**" tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`) to enhance its generation capabilities beyond simple left-to-right completion.

*   **Phi-3 as a Smaller, Performant Model:** The book frequently uses **Phi-3-mini** as a primary example of a generative model due to its relatively small size (**3.8 billion parameters**) and high performance, allowing it to run on devices with less VRAM. Phi-3 reuses the tokenizer of **Llama 2** but adds several special chat tokens like `<|user|>`, `<|assistant|>`, and `<|system|>` to better handle conversational contexts. The source emphasizes that the code examples in the book are designed to work with various LLMs, including Phi-3, and directs readers to resources like the Hugging Face site to access these models and their tokenizers.

*   **Generative LLMs as Autocompletion and Chatbots:** Generative LLMs like those in the GPT family function as **sequence-to-sequence machines** that **autocomplete text**. Their power is further unlocked by **fine-tuning** them to become **chatbots** capable of answering questions and following instructions.

*   **Transformer Architecture Underpinning:** The GPT family, including GPT-1 and presumably subsequent versions, is built upon the **Transformer architecture**, which utilizes **attention mechanisms** to understand and generate text by attending to different parts of the input sequence. This architecture is crucial to their ability to process context and generate coherent and relevant outputs.

*   **Evaluation of Generative Models:** The sources mention that generative models, including the GPT family, are evaluated using various **benchmarks** such as **MMLU, GLUE, TruthfulQA, and GSM8k** to assess their language understanding and generation abilities.

In summary, the sources present the **GPT family as quintessential examples of decoder-only generative models** that have significantly advanced the field of Language AI. From the foundational GPT-1 to the more recent and powerful GPT-4 and the efficient Phi-3, these models demonstrate the power of the decoder-only Transformer architecture for text generation and have become central to applications like chatbots and various other language-based tasks. The ongoing development and scaling of these models continue to define what is possible with generative AI.

####  Llama family (Llama 2)

These sources discuss the **Llama 2 family** as a prominent example within the larger context of **decoder-only generative models**. Here's what the sources say:

*   **Decoder-Only Architecture:** Llama 2, like the GPT family, utilizes a **decoder-only architecture**. This architecture is designed for **generative tasks**, where the model takes in text and attempts to **autocomplete it**. These models function as **sequence-to-sequence machines** and are often referred to as **completion models**.

*   **Relationship to LLMs:** Generative models with decoder-only architectures, including Llama 2, are commonly referred to as **Large Language Models (LLMs)**, especially the larger variants. While the term LLM can also encompass representation models, generative models are often the first type that comes to mind.

*   **Examples and Context:** The book mentions **Llama 2 alongside Phi-3** when comparing trained tokenizers. This implies that Llama 2 is a significant and relevant generative model for comparison. The release of models like Llama 2 contributed to what the book refers to as the "**Year of Generative AI**".

*   **Tokenizer:** The tokenizer used by **Phi-3 reuses the tokenizer of Llama 2** but adds several special chat tokens. This highlights a connection and potential similarities in the tokenization approach between these two decoder-only models. The Llama 2 tokenizer, like Phi-3's, includes special tokens like `<|user|>`, `<|assistant|>`, and `<|system|>` to better handle **conversational contexts**.

*   **Attention Mechanism:** Llama 2 employs **grouped-query attention**, an efficient tweak to the Transformer architecture, for its attention mechanism. This is mentioned as a more recent development to improve the inference scalability of larger models.

*   **Training Data:** The creation of the Llama 2 family involved training on a massive dataset containing **2 trillion tokens**. This underscores the vast amounts of data and computational resources required to train such large generative models. Meta used A100-80 GB GPUs for training, with estimated costs exceeding $5,000,000.

*   **Open Source Nature:** While some powerful LLMs are proprietary (closed source), Llama 2 is presented as an **open foundation and fine-tuned chat model**. To use Llama 2, a **Hugging Face account is required** to download the model after creating a free account and access token. The open-source nature allows users to have complete control over the model, enabling local usage, fine-tuning, and transparency.

*   **Choosing a Text Generation Model:** Llama 2 is mentioned as an option when **choosing a text generation model**, sitting alongside other models like Phi-3, Mistral, and StableLM. The book generally focuses more on open-source models like Llama 2 and Phi-3 due to their flexibility and free usage.

*   **Reward Models:** Llama 2 is noted for training **two reward models** for preference tuning: one scoring helpfulness and another scoring safety. This highlights the advanced techniques used to align these generative models with desired human preferences.

In summary, the sources position **Llama 2 as a significant open-source decoder-only generative model** within the broader landscape of Language AI. It is characterized by its decoder-only Transformer architecture, large-scale training, specific tokenizer with chat-focused tokens, efficient attention mechanism, and the use of reward models for fine-tuning. Its open nature makes it a key model discussed in the context of using and understanding LLMs.

---

### Representation models (encoder-only)

These sources discuss **representation models (encoder-only)** as a significant category within the larger context of **Language Models (LMs)**. Here's a breakdown of what the sources say about them:

*   **Definition and Purpose:** The sources clearly distinguish **representation models** from **generative models**. Representation models, built using an **encoder-only architecture**, primarily focus on **representing language**. Their main function is often to create **embeddings**, which are numerical vector representations of text. These embeddings capture the semantic and contextual nature of the input. The teal color and a small vector icon are used throughout the book to visually identify representation models.

*   **Architecture:** Representation models like **BERT (Bidirectional Encoder Representations from Transformers)** utilize a stack of **encoder blocks**. These blocks consist of **self-attention** mechanisms and **feedforward neural networks**. A special **[CLS] (classification) token** is often added to the input, and its final embedding is used as the representation for the entire input sequence, particularly for classification tasks. Figure 1-21 illustrates the architecture of a BERT base model with 12 encoders.

*   **Examples:** **BERT** is presented as a foundational example of an encoder-only architecture. The sources also mention various **BERT-like models**, including **RoBERTa, DistilBERT, ALBERT, and DeBERTa**. These models are highlighted as popular choices for creating task-specific and embedding models.

*   **Training:** Training encoder stacks, as in BERT, often involves a technique called **masked language modeling**. As shown in Figure 1-22, this method masks a part of the input, and the model is trained to predict the masked tokens. This training procedure allows BERT and related architectures to create accurate contextual language representations. These models are often **pretrained** on large corpora like Wikipedia to learn the semantic and contextual nature of text. Subsequently, they can be **fine-tuned** for specific downstream tasks like text classification.

*   **Use Cases:** Representation models are commonly used for **transfer learning**, where a pretrained model is adapted for a specific task. The sources highlight several use cases:
    *   **Text Classification:** Encoder-only models like BERT excel in task-specific use cases for classification. They can be fine-tuned directly for sentiment analysis or other classification tasks.
    *   **Generating Embeddings:** Representation models can be used to generate general-purpose embeddings that can be leveraged for various tasks beyond classification, such as semantic search.
    *   **Semantic Search:** Embedding models derived from encoder-only architectures are crucial for building semantic search systems.
    *   **Named-Entity Recognition (NER):** Representation models can be fine-tuned for word-level classification tasks like NER.

*   **Distinction from Generative Models:** The sources emphasize the fundamental difference between representation and generative models. While representation models primarily focus on understanding and encoding the input text into meaningful embeddings, **generative models (decoder-only)** focus on **generating text**. Representation models typically do not generate text as their primary function. The pink color and a small chat icon are used to visually identify generative models.

*   **Role in the Larger Context of LMs:** Although generative models might be the first thing people think of when discussing LLMs, representation models are still highly valuable in the field of Language AI. The book uses the term "large language models" loosely, acknowledging that it's often associated with generative decoder-only models, but also includes representation models, especially considering that "large" is an arbitrary and evolving description. Even smaller representation models (with fewer than 1 billion parameters) that don't generate text are considered within the scope of "large language models" in the context of this book. Representation models and the embeddings they produce are crucial components that can even **empower LLMs** in various applications like retrieval-augmented generation.

In summary, these sources present representation models (encoder-only) as a vital branch of Language Models, distinct from generative models in their primary focus on understanding and representing language through embeddings. Models like BERT and its variants have significantly impacted various language understanding tasks and continue to be essential tools in the Language AI landscape.

####  BERT family (BERT, ALBERT, DistilBERT, RoBERTa, DeBERTa)

These sources provide significant information about the **BERT family** (BERT, ALBERT, DistilBERT, RoBERTa, DeBERTa) within the larger context of **representation models (encoder-only)**.

**Representation Models (Encoder-Only):**

*   These models, in contrast to generative models, primarily focus on **representing language**. Their main function is often to create **embeddings**, which are numerical vector representations of text that capture semantic and contextual meaning. The sources visually identify representation models with a teal color and a small vector icon.
*   Representation models typically utilize an **encoder-only architecture**, meaning they only use the encoder part of the original Transformer architecture and remove the decoder entirely.

**BERT as the Foundation:**

*   **BERT (Bidirectional Encoder Representations from Transformers)**, introduced in 2018, is presented as a foundational encoder-only model that has been crucial in Language AI. Its architecture consists of **stacked encoder blocks**, which include **self-attention** mechanisms and **feedforward neural networks**. Figure 1-21 illustrates the architecture of a BERT base model with 12 encoders.
*   BERT's input often includes a special **[CLS] (classification) token**, and its final embedding is frequently used as the representation for the entire input sequence, especially for classification tasks.
*   BERT models are trained using **masked language modeling (MLM)**, where a portion of the input is masked, and the model learns to predict the masked tokens. This training allows BERT to create more accurate contextual representations of language. Figure 1-22 illustrates this training process.
*   This architecture and training procedure make BERT and related models excellent at representing **contextual language**. They are commonly used for **transfer learning**, where they are first pretrained on large datasets like Wikipedia to understand the semantic and contextual nature of text and then **fine-tuned** for specific downstream tasks, such as text classification. Figures 1-23 and 4-3 illustrate this fine-tuning process.
*   A significant benefit of pretrained BERT models is that most of the training is already done, making fine-tuning less computationally intensive and requiring less data. BERT models can also be used as **feature extraction machines** to generate embeddings without the need for fine-tuning.

**The BERT Family:**

*   Over the years, many variations of BERT have been developed, including **RoBERTa, DistilBERT, ALBERT, and DeBERTa**. These are referred to as **BERT-like models** and are popular choices for creating task-specific and embedding models. Figure 4-5 provides a timeline of common BERT-like model releases.
*   These models often build upon the core BERT architecture with various optimizations and training strategies. For example, **DistilBERT** is a smaller, faster, cheaper, and lighter distilled version of BERT.

**Role within Representation Models:**

*   Encoder-only models like the BERT family are central to the concept of representation models. They are primarily used for tasks that require understanding and encoding the input text into meaningful embeddings.
*   These models are crucial for various common tasks, including **classification tasks** (sentiment analysis, etc.), **clustering tasks**, and **semantic search**.
*   The embeddings generated by BERT and its family can also be used in more complex applications like **retrieval-augmented generation (RAG)**.
*   The sources also discuss using BERT for **named-entity recognition (NER)**, a token-level classification task. Fine-tuning BERT for NER involves making predictions for individual tokens in a sequence rather than the entire document.

**Distinction from Generative Models:**

*   The sources consistently differentiate the BERT family (and other encoder-only models) from **generative models (decoder-only)** like the GPT family. While representation models focus on understanding and encoding text, generative models focus on generating text.

In summary, the BERT family represents a cornerstone within the larger context of representation models (encoder-only). BERT's innovative architecture and training method paved the way for numerous subsequent models like RoBERTa, DistilBERT, ALBERT, and DeBERTa, all designed for effectively learning contextual language representations. These models and the embeddings they produce are fundamental for a wide range of language understanding tasks and continue to be highly influential in the field of Language AI.

---

### Encoder-decoder models

The sources discuss encoder-decoder models primarily within the context of the **Transformer architecture**, which was the original design consisting of both an encoding and a decoding component stacked on top of each other. This architecture is highlighted as being well-suited for **translation tasks**.

Key aspects of encoder-decoder models as described in the sources include:

*   **Architecture:** The original Transformer model combines stacked encoder and decoder blocks, where the input flows through each encoder and decoder. In contrast to encoder-only models like BERT and decoder-only models like GPT, the encoder-decoder architecture utilizes both components.
*   **Attention Mechanism:** Both the encoder and decoder blocks in the Transformer revolve around the attention mechanism instead of relying on Recurrent Neural Networks (RNNs) with attention features.
*   **Autoregressive Decoding:** Similar to decoder-only models, the decoder in encoder-decoder models is autoregressive, meaning it needs to consume each generated word before creating a new word.
*   **Input Processing:** The encoder's role is to represent the input as effectively as possible, generating a context in the form of an embedding that serves as the input for the decoder. The decoder then uses this representation to generate the output sequence.
*   **Task Specificity of Original Transformer:** While effective for translation, the original encoder-decoder Transformer architecture could not be easily used for other tasks like text classification. This limitation led to the development of encoder-only and decoder-only architectures tailored for different purposes.

The sources specifically mention the **Text-to-Text Transfer Transformer (T5)** as an interesting family of language models that leverage the encoder-decoder architecture.

*   **T5 Architecture:** T5's architecture is similar to the original Transformer, with a stack of encoders and decoders (e.g., 12 of each).
*   **Pretraining of T5:** These models were first pretrained using a masked language modeling technique where sets of tokens (or token spans) were masked during pretraining, rather than individual tokens.
*   **Fine-tuning of T5:** The pretraining method was extended by fine-tuning on a large variety of tasks that closely follow instructions, leading to the Flan-T5 family of models.
*   **T5 for Classification:** Despite the rise of encoder-only (representation) and decoder-only (generative) models, encoder-decoder models like T5 can also be used for text classification. This is achieved by framing the classification task as a text generation task.

In the broader context of Language Models:

*   Encoder-decoder models represent one of the primary architectural designs within the Transformer family.
*   They are distinct from **encoder-only models** (like BERT), which focus on representing language and generating embeddings for tasks such as classification and semantic search. Encoder-only models remove the decoder entirely.
*   They are also different from **decoder-only models** (like GPT), which primarily focus on generating text by taking in some text and attempting to autocomplete it. Decoder-only models stack decoder blocks and remove the encoder-attention block.
*   While the original Transformer encoder-decoder was primarily for sequence-to-sequence tasks like translation, models like T5 demonstrate the versatility of this architecture for other generative tasks, including text classification, by reformulating problems as text generation.

Therefore, encoder-decoder models, particularly those based on the Transformer architecture like T5, play a significant role in the landscape of Language Models, offering a different set of capabilities and strengths compared to their encoder-only and decoder-only counterparts.

* **Flan-T5**
* **BLIP-2**

#### Flan-T5

The sources discuss Flan-T5 as an **interesting family of language models that leverage the encoder-decoder architecture**. It is explicitly mentioned as an example of an encoder-decoder model, contrasting with encoder-only models like BERT and decoder-only models like ChatGPT.

Here's what the sources say about Flan-T5 in the context of encoder-decoder models:

*   **Architecture:** Flan-T5's architecture is described as **similar to the original Transformer**, consisting of a stack of encoders and decoders. Specifically, the example Flan-T5 model mentioned has **12 decoders and 12 encoders stacked together**. This is visually represented in Figure 4-19, which illustrates the decoder-encoder architecture.

*   **Pretraining:** These models were initially **pretrained using a masked language modeling technique**. However, unlike some other models that mask individual tokens, **Flan-T5's pretraining involved masking sets of tokens (or token spans)**. This pretraining step is illustrated in Figure 4-20.

*   **Fine-tuning:** The sources highlight the **unique fine-tuning approach** used for Flan-T5. Instead of fine-tuning for a single specific task, **each task is converted into a sequence-to-sequence task and the model is trained on a wide variety of tasks simultaneously**. This is shown in Figure 4-21, which depicts the conversion of specific tasks into textual instructions for training. The Flan-T5 family further extends this by fine-tuning on **more than a thousand tasks that closely follow instructions**, similar to those used with GPT models.

*   **Generative Nature:** Like decoder-only models, Flan-T5, being an encoder-decoder architecture used as a sequence-to-sequence model, generally falls into the category of **generative models**.

*   **Application in Classification:** The sources demonstrate how Flan-T5, despite its encoder-decoder architecture, can be effectively used for **text classification**. This is achieved by **instructing the model** to output the sentiment of a given text. The example provided shows Flan-T5 achieving a respectable F1 score of 0.84 on a sentiment analysis task, highlighting the capabilities of generative models for classification. This contrasts with the traditional use of encoder-decoder models primarily for tasks like translation.

*   **Tokenizer:** Flan-T5 uses a tokenizer implementation called **SentencePiece**, which supports Byte Pair Encoding (BPE) and the unigram language model. The example tokenized text for Flan-T5 shows how it handles capitalization and out-of-vocabulary tokens. Notably, the Flan-T5 tokenizer **does not have newline or whitespace tokens**, which could be a limitation for tasks involving code.

In the larger context of Encoder-decoder models:

*   Flan-T5 exemplifies how the original Transformer's encoder-decoder architecture can be adapted and utilized for a broader range of tasks beyond its initial focus on translation.
*   It distinguishes itself through its **pretraining on masked spans** and its **instruction-based, multi-task fine-tuning**, which allows it to perform well on diverse tasks by framing them as text-to-text problems.
*   The success of Flan-T5 demonstrates that while encoder-only models excel at representation learning and decoder-only models at open-ended generation, encoder-decoder models can also be powerful **generative models capable of following instructions and performing tasks like classification** when appropriately trained and prompted.

In summary, Flan-T5 represents a significant development in encoder-decoder models by showcasing their versatility through innovative pretraining and fine-tuning strategies, enabling them to perform effectively on a wide array of tasks, including those traditionally associated with encoder-only or decoder-only architectures.

#### BLIP-2

These sources discuss BLIP-2 primarily within the context of **multimodal large language models (LLMs)**, rather than directly as an encoder-decoder model in the same vein as the original Transformer or T5. However, it does leverage aspects of both encoding and decoding and bridges different modalities, which indirectly relates to the functionalities often associated with encoder-decoder architectures.

Here's what the sources say about BLIP-2:

*   **Multimodal Capability:** BLIP-2 is highlighted as a method for introducing **vision capabilities to existing language models**, enabling them to reason about both text and images. Models that can handle multiple types of data like text and images are called multimodal. BLIP-2 allows an LLM to, for example, look at a picture and answer questions about it.

*   **Bridging the Modality Gap:** Creating a multimodal LLM from scratch requires significant resources. Instead, BLIP-2 offers an **easy-to-use and modular technique** to bridge the gap between vision and language by using a component called the **Querying Transformer (Q-Former)**.

*   **Leveraging Pretrained Models:** A key aspect of BLIP-2 is that it **connects a pretrained image encoder (Vision Transformer - ViT) and a pretrained LLM**. By doing so, it **only needs to train the Q-Former** without needing to train the image encoder and LLM from scratch, making efficient use of existing technology.

*   **Training Process:** The Q-Former is trained on three tasks:
    *   **Image-text contrastive learning:** Aligning pairs of image and text embeddings.
    *   **Image-text matching:** Predicting whether an image and text pair is matched or unmatched.
    *   **Image-grounded text generation:** Training the model to generate text based on information from the input image.
    These tasks are jointly optimized to improve the visual representations extracted from the frozen ViT. The goal is to **inject textual information into the embeddings of the frozen ViT** so they can be used in the LLM.

*   **Soft Visual Prompts:** The learnable embeddings derived from the Q-Former (which now contain visual information) are passed to the LLM. These embeddings act as **soft visual prompts** that condition the LLM on the visual representations extracted by the Q-Former. A fully connected linear layer ensures these embeddings have the same shape the LLM expects.

*   **Relation to LLMs:** BLIP-2 demonstrates a method to extend the capabilities of existing **textual LLMs** to handle visual inputs. Many other visual LLMs with similar processes, like LLaVA and Idefics 2, have been released since BLIP-2, all aiming to **project visual features to language embeddings** that can be used as input for an LLM.

*   **Use Cases:** The sources mention several interesting use cases for BLIP-2, including **captioning images**, **answering visual questions**, and even **performing prompting** that combines both image and text. An example of image captioning shows BLIP-2 generating the caption "an orange supercar driving on the road at sunset" from an image. Another use case demonstrates multimodal chat-based prompting where the model can reason about input images.

In the larger context of Encoder-decoder models:

While BLIP-2 isn't strictly an encoder-decoder in the traditional sense of the Transformer architecture for tasks like translation, it incorporates aspects of both. The **pretrained image encoder (ViT) can be seen as an encoder** that processes visual information into embeddings. The **pretrained LLM acts as a decoder** that generates textual output based on these visual embeddings (mediated by the Q-Former) and potentially textual input.

The Q-Former acts as a bridge between these modalities, learning to extract relevant information from the visual encoder and format it in a way that the language decoder can understand. This bridging function can be conceptually linked to the role of the encoder in providing a meaningful context vector for the decoder in traditional encoder-decoder models.

Therefore, BLIP-2, and similar multimodal LLMs, represent an **extension of the core ideas behind encoder-decoder models** by applying them across different data modalities. Instead of encoding text to a latent space for decoding into another text sequence (like in translation), they encode images to a latent space (via the Q-Former) that can be used by a language model to decode into text. This highlights the **adaptability of the underlying principles of encoding information and then decoding it to generate relevant outputs**, even when the input and output modalities differ.

---

### Training

The sources provide a comprehensive overview of the training paradigms for language models (LMs), particularly large language models (LLMs), within the broader context of Language AI. The book emphasizes that creating capable LMs typically involves at least two key stages: **pretraining** and **fine-tuning**.

**Pretraining**:

*   This initial stage is the most computationally intensive and time-consuming. During pretraining, an LM is trained on a **vast corpus of internet text**. The goal is to enable the model to learn fundamental aspects of language, including grammar, context, and language patterns.
*   Pretraining is often a **self-supervised** process where the model learns to predict the next token in a sequence. For representation models like **BERT (encoder-only)**, a technique called **masked language modeling (MLM)** is used, where parts of the input are masked, and the model is trained to predict the masked tokens. This allows BERT to create accurate contextual representations of the input. For **generative models (decoder-only)** like the **GPT family**, the pretraining involves predicting the next word to autocomplete text.
*   The model resulting from pretraining is often referred to as a **foundation model** or **base model**. These base models have a broad understanding of language but generally **do not follow specific instructions**. Examples of datasets used for pretraining include large collections of web pages (Common Crawl) and books. The size of the model, indicated by the number of parameters, significantly influences its capabilities.

**Fine-tuning**:

*   The second crucial step is **fine-tuning** (or post-training), where a pretrained model is further trained on a **narrower task** or to exhibit desired behavior. This allows the LLM to adapt to specific applications such as text classification, sentiment analysis, or following instructions.
*   Fine-tuning is more resource-efficient than pretraining because it leverages the knowledge already acquired during the initial phase. This process can involve training on task-specific datasets.
*   The book discusses different strategies for fine-tuning generative models, including **supervised fine-tuning (SFT)**, where the model is trained on instruction data to produce desired outputs, and **preference tuning**, where models are further refined based on human preferences, as seen in the training of ChatGPT.
*   For representation models used in tasks like classification, fine-tuning often involves adding a task-specific layer on top of the pretrained encoder and training the combined model on labeled data. **Transfer learning**, where a model pretrained on a large general dataset is adapted for a specific task, is a common approach.

**Training of Embedding Models**:

*   The sources also detail the training of **text embedding models**, which are crucial for tasks like semantic search and text clustering. A key technique discussed is **contrastive learning**. This approach trains the model to understand similarity and dissimilarity between documents by feeding it examples of similar and dissimilar pairs. Word2vec, an earlier embedding method, also utilized contrastive learning principles.
*   Fine-tuning pretrained language models can also be used to create high-quality **contextualized word embeddings**.

**Ethical Considerations in Training**:

*   The book highlights that LLMs are trained on large amounts of data that may contain **biases**. As a result, LLMs might learn and reproduce these biases, potentially amplifying them. The training data is often not shared, making it difficult to understand potential biases without testing the models.

In the larger context of Language Models:

*   The two-stage training paradigm (pretraining and fine-tuning) has become a standard for developing high-performing LMs. Pretraining allows models to learn a broad understanding of language from massive datasets, while fine-tuning adapts these foundational models to specific downstream tasks.
*   The distinction between training objectives for different model architectures (MLM for encoder-only, next-token prediction for decoder-only) reflects the intended use of these models – representation learning versus text generation.
*   The development of techniques like contrastive learning for training embedding models showcases the ongoing research into optimizing language representation for various applications beyond just text generation.
*   The awareness of potential biases in training data underscores the importance of ethical considerations in the development and deployment of LMs.

Overall, the sources emphasize the multi-stage and resource-intensive nature of training LLMs, highlighting the importance of both broad pretraining and task-specific fine-tuning in creating versatile and capable language models. They also touch upon the specialized training for different types of language models and the ethical implications related to the data used for training.

* **Pre-training (masked language modeling)**
* **Fine-tuning (task-specific)**
* **Instruction tuning**
* **Preference tuning (DPO)**


#### Pre-training (masked language modeling)

The sources discuss **masked language modeling (MLM)** primarily as a **key technique used in the pretraining stage** of certain language models, particularly **representation models like BERT**. In the larger context of training language models, MLM plays a significant role in enabling these models to learn rich contextual representations of language.

Here's what the sources say about pre-training with MLM:

*   **Pretraining as a Foundational Step:** Creating large language models (LLMs) typically involves at least two steps: **pretraining and fine-tuning**. **Pretraining is the first and most computationally intensive step**, where a language model is trained on a vast corpus of internet text. The goal of pretraining is to enable the model to learn fundamental aspects of language, including grammar, context, and language patterns. Models that undergo pretraining are considered **pretrained models** or **foundation models**.

*   **MLM for Representation Models:** **MLM is a self-supervised training technique adopted by models like BERT during pretraining**. As illustrated in Figure 1-22, this method **masks a part of the input text, and the model is trained to predict the masked tokens**. This task is difficult but allows the model to create **more accurate contextual representations of the input**.

*   **Learning Contextual Language:** The architecture and training procedure using MLM make BERT and related architectures **incredible at representing contextual language**. By training on a massive dataset like the entirety of Wikipedia, BERT learns to understand the **semantic and contextual nature of text**.

*   **Transfer Learning:** BERT-like models trained with MLM are commonly used for **transfer learning**. This involves **first pretraining the model for language modeling (using MLM) and then fine-tuning it for a specific downstream task**, such as text classification.

*   **Feature Extraction:** BERT models generate embeddings at almost every step in their architecture. This makes BERT models useful as **feature extraction machines even without fine-tuning** on a specific task.

*   **Contrast to Generative Pretraining:** While BERT uses MLM for pretraining, **generative models (decoder-only) like the GPT family use next-token prediction** as their pretraining objective. The goal for generative models is to take in some text and attempt to autocomplete it.

*   **Continued Pretraining:** The sources also discuss the concept of **continued pretraining**. Instead of just a two-step process of pretraining and fine-tuning, an additional step can be added: **continuing to pretrain an already pretrained BERT model on domain-specific data using MLM**. This updates the subword representations to be more attuned to words the model might not have seen before, potentially improving performance on downstream tasks like classification in that specific domain.

In summary, within the larger context of language model training, **pretraining with masked language modeling is a crucial step for encoder-only (representation) models like BERT**. It enables these models to learn deep contextual understanding of language by predicting masked words in vast amounts of text. This pretrained knowledge can then be effectively leveraged through fine-tuning for various downstream tasks, making MLM a cornerstone of modern natural language processing and transfer learning.

#### Fine-tuning (task-specific)

The sources explain that **fine-tuning is a crucial second step in the training paradigm of large language models (LLMs), following the initial phase of pretraining**. In contrast to traditional machine learning, which generally involves a single step of training a model for a specific task, creating LLMs typically requires this two-step approach.

**The primary purpose of task-specific fine-tuning is to take a broadly pretrained foundation model and adapt it to perform well on a narrower task or to exhibit desired behavior**. Pretrained base models, having learned grammar, context, and language patterns from vast amounts of internet text, generally do not follow specific instructions. Fine-tuning allows these models to be tailored for applications like text classification, sentiment analysis, generating specific types of content, or functioning as chatbots.

The sources highlight several aspects and methods related to task-specific fine-tuning:

*   **Resource Efficiency:** Fine-tuning is generally **less compute-intensive and requires less data** compared to the initial pretraining phase. This is because it leverages the knowledge already acquired during pretraining. The majority of the heavy lifting in terms of learning fundamental language representations has already been done.

*   **Supervised Fine-Tuning (SFT):** This is a common method where a pretrained model is further trained on a **smaller but labeled dataset specific to the target task**. For generative models, SFT often involves training on instruction data (user queries with corresponding desired answers) to make the model follow instructions. For representation models like BERT, fine-tuning for classification involves training on labeled examples of text and their corresponding classes.

*   **Parameter-Efficient Fine-Tuning (PEFT):** To address the computational cost and storage requirements of updating all model parameters during full fine-tuning, PEFT techniques have emerged. These methods focus on fine-tuning pretrained models with higher computational efficiency by only updating a small subset of parameters. Examples discussed include:
    *   **Adapters:** These involve adding small, modular components within the Transformer architecture that are fine-tuned for a specific task, while keeping the original model weights frozen. This can achieve performance comparable to full fine-tuning by only training a small percentage of the parameters.
    *   **Low-Rank Adaptation (LoRA):** LoRA works by adding low-rank matrices to the original weight matrices and only training these low-rank matrices during fine-tuning. This significantly reduces the number of trainable parameters, leading to faster training and lower memory usage. QLoRA extends LoRA by incorporating quantization to further reduce memory constraints.

*   **Preference Tuning (Alignment):** For generative models intended for interactive use like chatbots, a further fine-tuning step called preference tuning or alignment is often employed after SFT. This process aligns the model's output with human preferences for helpfulness, safety, and other desirable qualities. This often involves training a reward model based on human-ranked outputs and then using this reward model to further fine-tune the LLM, for example, using Direct Preference Optimization (DPO).

*   **Fine-tuning for Different Model Architectures:** The specific approach to fine-tuning can differ based on the underlying model architecture. Encoder-only models like BERT are often fine-tuned by adding a task-specific layer on top for classification or other representation-based tasks. Decoder-only models like the GPT family are fine-tuned for text generation, instruction following, and chatbot functionalities by training them to predict the next token given specific prompts or instructions. Encoder-decoder models like T5 can be fine-tuned for various sequence-to-sequence tasks by framing each task as a text-to-text problem.

*   **Continued Pretraining:** The sources also mention the possibility of an intermediate step before task-specific fine-tuning, called continued pretraining. This involves taking an already pretrained model and further training it on domain-specific data using the original pretraining objective (like masked language modeling for BERT). This can help the model better adapt to the specific vocabulary and nuances of a particular domain before being fine-tuned for a downstream task, potentially improving performance.

*   **Creating Embedding Models:** Fine-tuning is also crucial for creating specialized text embedding models. While some embedding models are trained from scratch using contrastive learning, pretrained language models can also be fine-tuned on data that emphasizes semantic similarity or other specific relationships to generate high-quality embeddings for tasks like semantic search and text clustering.

In conclusion, the sources position task-specific fine-tuning as an indispensable step in leveraging the power of pretrained language models. It allows for efficient adaptation of these foundation models to a wide array of practical applications and desired behaviors through various techniques tailored to the specific task and model architecture. This multi-step training paradigm has become the standard for developing capable and versatile language AI systems.

#### Instruction tuning

The sources discuss **instruction tuning as a crucial step, often referred to as supervised fine-tuning (SFT), in the larger context of training high-quality large language models (LLMs)**. This step follows the initial **pretraining phase**, where a base model learns general language representations by predicting the next token on massive datasets.

Here's a breakdown of what the sources say about instruction tuning:

*   **Position in the Training Paradigm:** The creation of high-quality LLMs typically involves three steps: **pretraining (language modeling), supervised fine-tuning (instruction tuning), and preference tuning (alignment)**. Instruction tuning is the **second step**, taking a pretrained foundation or base model and further training it.

*   **Purpose of Instruction Tuning:** The primary goal of instruction tuning is to **make LLMs more useful by enabling them to respond well to instructions and follow them**. Base models, trained only on next-token prediction, often attempt to complete input phrases rather than answering questions or following directions. Instruction tuning adapts these models to understand and execute user instructions. For example, a base model might try to complete the question "Write an article about LLMs" by listing more instructions, whereas an instruction-tuned model would generate the article.

*   **Methodology: Supervised Learning:** Instruction tuning is a form of **supervised learning**. It involves training the pretrained model on a **smaller but labeled dataset**. This labeled data consists of **question-response pairs or instruction data**, where each user query (instruction) is paired with the desired correct answer or output. During training, the model takes the input instruction and uses next-token prediction on the target output (response).

*   **Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning (PEFT):** The sources discuss different ways to perform instruction tuning:
    *   **Full Fine-Tuning:** This involves **updating all the parameters of the model** using the instruction dataset. While potentially leading to high performance, it can be **costly, time-consuming, and require significant storage**.
    *   **Parameter-Efficient Fine-Tuning (PEFT):** To address the limitations of full fine-tuning, PEFT techniques have gained prominence. These methods aim to fine-tune pretrained models with **higher computational efficiency by only updating a small subset of parameters**. The source specifically mentions **Low-Rank Adaptation (LoRA)** and its quantized version **QLoRA** as examples of PEFT techniques used for instruction tuning. LoRA works by adding low-rank matrices to the original weight matrices and only training these smaller matrices. QLoRA further enhances efficiency by incorporating quantization to reduce memory usage. Instruction tuning with QLoRA allows for fine-tuning large models on instruction data with reduced computational resources.

*   **Benefits of Instruction Tuning:** Instruction tuning is crucial for bridging the gap between a base language model and a model that is readily usable for various applications. It enables the model to exhibit **chat-like behavior and closely follow instructions**, making it more aligned with user expectations.

*   **Relation to Other Fine-Tuning Steps:** Instruction tuning is often followed by **preference tuning (or alignment)**, which further refines the model's behavior to align with human preferences for helpfulness, safety, and other desirable qualities. This involves training the model on data that ranks different model outputs based on human preferences.

In essence, instruction tuning is a critical intermediate step in the training of modern LLMs. It leverages the broad language understanding learned during pretraining and focuses it on the ability to follow specific instructions, thereby transforming a general-purpose language model into a more controllable and application-ready AI system.

#### Preference tuning (DPO)

The sources describe **Direct Preference Optimization (DPO) as a method for preference tuning, which is the final step in creating high-quality large language models (LLMs)**, following **pretraining (language modeling)** and **supervised fine-tuning (SFT) or instruction tuning**.

In the larger context of training, the sources outline a three-step process:

1.  **Pretraining (Language Modeling):** This initial phase involves training an LLM on massive text datasets to learn linguistic and semantic representations by predicting the next token. This results in a base or foundation model that understands language but doesn't necessarily follow instructions well.

2.  **Supervised Fine-Tuning (SFT) / Instruction Tuning:** This second step takes the pretrained model and further trains it on a smaller, labeled dataset of instruction-response pairs. The goal is to make the LLM respond well to instructions and exhibit chat-like behavior. Techniques like Parameter-Efficient Fine-Tuning (PEFT) such as LoRA and QLoRA can be used during this stage to reduce computational costs.

3.  **Preference Tuning / Alignment:** This final step further improves the model's quality by aligning its output with human preferences for helpfulness, safety, and other desirable qualities. **DPO is presented as an alternative to Reinforcement Learning from Human Feedback (RLHF) methods like Proximal Policy Optimization (PPO) for achieving this alignment**.

Here's what the sources specifically say about DPO within this context:

*   **Purpose:** DPO aims to align the LLM's output with human preferences by optimizing the likelihood of accepted generations over rejected generations. This is done by training the model to understand what constitutes a better response compared to a less desirable one.

*   **How it Works:** Instead of using a separate reward model to judge the quality of a generation (as in PPO), **DPO utilizes the LLM itself as a reward model by comparing the output of a frozen reference model with that of the trainable model**. The log probabilities of rejected and accepted generations are extracted from both models at the token level. By calculating the shift in these probabilities, the training process optimizes the trainable model to be more confident in generating accepted responses and less confident in generating rejected ones.

*   **Advantages:** The sources highlight that **DPO does away with the reinforcement-based learning procedure used in PPO, making it a more stable and potentially more accurate method for preference tuning**. It can also simplify the training process by potentially removing the need to train a separate reward model. Furthermore, DPO can be combined with PEFT techniques like LoRA for efficient preference tuning.

*   **Implementation:** The sources provide examples of how to perform preference tuning with DPO using the Hugging Face `trl` library, often following an initial instruction tuning phase. This involves configuring a `DPOConfig` with training parameters and utilizing LoRA for parameter-efficient adaptation.

*   **Relation to SFT:** The sources mention that a combination of SFT followed by DPO is a common approach. SFT first fine-tunes the model for basic chatting and instruction following, and then DPO further aligns its answers with human preferences. Newer methods like Odds Ratio Preference Optimization (ORPO) even aim to combine SFT and DPO into a single training process.

In summary, the sources position DPO as a key technique in the final stage of LLM training, focused on aligning the model's behavior with human preferences in a more stable and potentially more efficient manner than traditional reinforcement learning approaches. It builds upon the foundation laid by pretraining and instruction tuning to create more useful and desirable language models.

---

### Attention mechanisms

The sources emphasize that **attention mechanisms are a crucial component driving the remarkable abilities of large language models (LLMs)**. The book positions the introduction of attention as a significant improvement over earlier recurrent neural network (RNN) architectures, particularly in handling the sequential nature of text and context.

Here's a breakdown of what the sources say about attention mechanisms in the context of language models:

*   **Encoding and Decoding Context with Attention:** The attention mechanism was initially introduced as a solution to improve upon RNN architectures by allowing a model to focus on the parts of the input sequence that are relevant to one another. It selectively determines which words are most important in a given sentence. For example, in the translation of "Ik hou van lama's" to Dutch, the attention mechanism of the decoder allows it to focus on "llamas" before generating "lama's". Instead of just passing a single context embedding, the hidden states of all input words are passed to the decoder, enabling the model to "attend" to the entire sentence.

*   **Attention Is All You Need: The Transformer Architecture:** The paper "Attention is all you need" introduced the **Transformer architecture, which is solely based on the attention mechanism**, removing the need for recurrence. This was a breakthrough because, compared to RNNs, Transformers could be trained in parallel, significantly speeding up training. Most models used throughout the book are Transformer-based.

*   **Self-Attention:** A key aspect of the Transformer is **self-attention**, which allows the model to attend to different positions within a single sequence, enabling a more accurate representation of the input. Unlike previous methods that processed one token at a time, self-attention can look at the entire sequence at once, both forward and backward (though in generative models, it's typically masked to only attend to previous tokens).

*   **How Attention Works:** The attention mechanism involves two main steps:
    *   **Relevance Scoring:** Scoring how relevant each of the previous input tokens is to the current token being processed. This is achieved by multiplying the query vector of the current position with the keys matrix, producing a relevance score for each previous token. These scores are then normalized using a softmax operation.
    *   **Combining Information:** Using the relevance scores to combine information from the various positions into a single output vector. This is done by multiplying the value vector associated with each token by its relevance score, and then summing up these resulting vectors.

*   **Multi-Head Attention:** To enhance the model's capacity to capture complex patterns, the attention mechanism is duplicated and executed multiple times in parallel. Each parallel application is called an **attention head**. This allows the model to attend to different types of information simultaneously.

*   **More Efficient Attention:** The attention layer is computationally expensive, leading to research on more efficient variations:
    *   **Local/Sparse Attention:** Limits the context of previous tokens the model can attend to, improving efficiency for larger models. GPT-3, for instance, interleaves full-attention and sparse-attention Transformer blocks.
    *   **Multi-Query and Grouped-Query Attention:** These techniques, used in models like Llama 2, improve inference scalability by reducing the size of the keys and values matrices involved in the attention calculation. Multi-query attention shares these matrices across all heads, while grouped-query attention shares them within groups of heads.
    *   **Flash Attention:** A method and implementation that significantly speeds up both training and inference of Transformer LLMs on GPUs by optimizing memory access during the attention calculation.

*   **Attention in Transformer Blocks:** A Transformer block consists of an attention layer (specifically a self-attention layer) followed by a feedforward neural network. The attention layer is primarily responsible for incorporating relevant information from other input tokens and positions into the representation of the current token.

*   **Masked Attention in Generative Models:** In generative (decoder-only) models, the attention mechanism is modified to prevent "looking into the future" by only allowing the model to attend to previous tokens during the generation process. This is crucial for the autoregressive nature of text generation, where each token is predicted based on the tokens generated before it.

*   **Attention in Representation Models (like BERT):** In contrast to generative models, encoder-only models like BERT utilize bidirectional attention, meaning they can attend to both preceding and succeeding tokens in a sequence. This allows them to build rich contextual representations of the entire input, which is beneficial for tasks like text classification.

In summary, the sources present **attention mechanisms as the foundational innovation behind the success of modern LLMs and the Transformer architecture**. It enables models to understand and generate language by effectively processing context, attending to relevant parts of the input, and allowing for parallel processing during training. Ongoing research continues to refine and optimize attention mechanisms for greater efficiency and capability.

* **Self-attention**
* **Multi-head attention**
* **Multi-query attention**
* **Grouped-query attention**
* **Local/sparse attention**
* **Flash Attention**


#### Self-attention

These sources highlight that **self-attention is a specific type of attention mechanism that is central to the architecture and capabilities of Transformer-based language models**. It's discussed as a significant advancement over earlier attention mechanisms used in recurrent neural networks (RNNs).

Here's what the sources say about self-attention in the larger context of attention mechanisms:

*   **Attention as a Foundation:** The book introduces attention as a solution to improve upon RNN architectures by enabling a model to focus on relevant parts of an input sequence. It allows the model to selectively determine which words are most important in a given sentence. Instead of just a single context embedding, the hidden states of all input words are passed to the decoder, allowing the model to "attend" to the entire sentence. The paper "Attention is all you need" then proposed the **Transformer architecture, which is solely based on the attention mechanism**, moving away from recurrence. This marked a significant shift and is credited with driving the "amazing abilities of large language models".

*   **Self-Attention's Key Feature:** **Self-attention distinguishes itself by allowing the model to attend to different positions within a *single* sequence**. This enables a more accurate representation of the input because, unlike processing one token at a time (as in RNNs), self-attention can look at the entire sequence simultaneously (though with masking in generative models). Figure 1-18 illustrates how self-attention can "look" both forward and back in a single sequence.

*   **How Self-Attention Works:** The process involves two main steps:
    1.  **Relevance Scoring:** Scoring how relevant each of the previous (or other, in bidirectional models) input tokens is to the current token being processed. This is achieved by multiplying the query vector of the current position with the keys matrix, producing relevance scores, which are then normalized using softmax.
    2.  **Combining Information:** Using these scores to combine information from various positions into a single output vector. This is done by multiplying the value vector of each token by its relevance score and then summing the results.

*   **Multi-Head Self-Attention:** To enhance the model's ability to capture complex patterns, the self-attention mechanism is duplicated and run in parallel multiple times, creating **multiple attention heads**. Each head can attend to different types of information simultaneously, increasing the model's capacity. Each head has its own distinct query, key, and value matrices.

*   **Self-Attention in Transformer Blocks:** A Transformer block consists of a **self-attention layer** followed by a feedforward neural network. The self-attention layer is primarily responsible for incorporating relevant information from other input tokens and positions into the representation of the current token.

*   **Masked Self-Attention in Generative Models:** In decoder-only (generative) models, the self-attention layer is modified to **mask future positions**. This ensures that during text generation, the model can only attend to previous tokens, preventing it from "looking into the future" and maintaining the autoregressive nature of the generation process. Figure 1-20 illustrates this concept.

*   **Bidirectional Self-Attention in Representation Models:** In contrast, encoder-only models like BERT utilize **bidirectional self-attention**, where the model can attend to both preceding and succeeding tokens in a sequence. This allows for building rich contextual representations of the entire input, which is beneficial for tasks like text classification.

*   **Efficiency Considerations:** The self-attention mechanism can be computationally expensive, leading to research on more efficient variations like **local/sparse attention** (limiting the context), **multi-query and grouped-query attention** (sharing key and value matrices to reduce size), and **Flash Attention** (optimizing memory access). GPT-3, for instance, interleaves full and sparse self-attention blocks. Llama 2 uses grouped-query attention.

In essence, self-attention is presented as the core mechanism within the Transformer architecture that allows language models to understand the relationships between different words in a sequence, regardless of their distance. This ability to weigh the importance of different parts of the input when processing or generating text is fundamental to the success of modern LLMs.

#### Multi-head attention

The sources emphasize that **multi-head attention is a crucial enhancement to the basic attention mechanism within the Transformer architecture**, significantly contributing to the ability of large language models (LLMs) to model complex patterns in language.

In the larger context of attention mechanisms, which were initially introduced to allow models to focus on relevant parts of an input sequence, multi-head attention expands this capability by running the attention mechanism in parallel multiple times. The sources explain this as follows:

*   The fundamental attention mechanism involves scoring the relevance of previous input tokens to the current token and then combining information based on these scores.
*   To give the Transformer "more extensive attention capability", this attention mechanism is **duplicated and executed multiple times in parallel**, with each parallel application being called an **attention head**. This is visually depicted in Figure 3-17.
*   The key benefit of multi-head attention is that it **increases the model's capacity to model complex patterns in the input sequence that require paying attention to different patterns at once**. Each head can learn to focus on different aspects of the relationships between the tokens.
*   Within each attention head, the calculation involves projecting the input token embeddings into **queries, keys, and values matrices** using individual projection matrices. The relevance scoring is done by multiplying the query of the current position with the keys matrix, and the information is combined by multiplying the value vectors by their respective relevance scores.
*   **In multi-head attention, each "attention head" has its own distinct query, key, and value matrices calculated for a given input**, as illustrated in Figure 3-26. This allows each head to learn different relationships in the data.
*   After the parallel attention operations in each head, their outputs are typically concatenated and then linearly transformed to produce the final output of the multi-head attention layer.
*   The source also discusses more recent efficient attention tweaks that build upon the concept of multi-head attention, such as **multi-query attention** and **grouped-query attention**. These methods aim to improve inference scalability by sharing the keys and values matrices either across all heads (multi-query) or within groups of heads (grouped-query), as shown in Figures 3-27 and 3-28. Models like Llama 2 utilize grouped-query attention.
*   The multi-head self-attention mechanism is a core component of the **encoder and decoder blocks** in the Transformer architecture. In the encoder, it helps to generate intermediate representations by attending to different positions within the input sequence. In the decoder of generative models, it is often used with masking to prevent attending to future tokens.

In summary, the sources present multi-head attention as a significant advancement over single-head attention, enabling Transformer-based LLMs to capture a richer understanding of language by attending to various relationships and patterns in the input simultaneously through parallel processing within multiple attention heads. This ability is considered a key reason behind the "amazing abilities of large language models".

#### Multi-query attention

These sources discuss **multi-query attention** as a more recent and **efficient optimization of the standard multi-head attention mechanism** within the Transformer architecture. It's presented in the larger context of attention mechanisms that were initially introduced to allow models to focus on relevant parts of an input sequence.

Here's what the sources say about multi-query attention:

*   **Optimization for Efficiency:** Multi-query attention is described as an "efficient attention tweak to the Transformer" that aims to improve the **inference scalability of larger models by reducing the size of the matrices involved**.

*   **Comparison to Multi-Head Attention:** The standard multi-head attention, as detailed in "The Illustrated Transformer" (not directly in these excerpts but referenced), involves each attention head having its own distinct query, key, and value matrices calculated for a given input. **Multi-query attention optimizes this by sharing the keys and values matrices between *all* the attention heads**. Therefore, in multi-query attention, the only unique matrices for each head are the queries matrices. Figure 3-27 illustrates this concept.

*   **Motivation:** As model sizes grow, the computational cost of attention becomes a significant factor. Multi-query attention is introduced as a way to mitigate this cost, specifically during inference, by reducing the memory footprint associated with the key and value projections.

*   **Building Block for Further Optimization:** The sources mention that **grouped-query attention** builds upon the concept of multi-query attention. Grouped-query attention represents a middle ground where keys and values matrices are shared within groups of attention heads, rather than across all heads, offering a trade-off between efficiency and model quality.

*   **Contrast with Other Attention Types:** Figure 3-25 provides a visual comparison of the original multi-head attention, grouped-query attention, and multi-query attention. This places multi-query attention within the spectrum of different approaches to implement the attention mechanism.

*   **Relationship to Model Architectures:** While the sources don't explicitly state which specific models use *only* multi-query attention, they mention that grouped-query attention, which builds on multi-query attention, is used by models like **Llama 2**.

In summary, the sources position multi-query attention as an **evolutionary step in attention mechanisms**, specifically designed to enhance the efficiency of large language models, particularly during inference. It achieves this by **sharing the key and value projection matrices across all attention heads**, contrasting with the independent projections in standard multi-head attention. This optimization contributes to the ongoing efforts to make large models more scalable and practical for deployment.

#### Grouped-query attention

These sources discuss **grouped-query attention** as a more recent **optimization of the multi-head attention mechanism** within the Transformer architecture, designed to improve the **inference scalability of large language models**. It is presented as an evolution in the landscape of attention mechanisms, building upon the concepts of standard multi-head attention and multi-query attention.

Here's how the sources describe grouped-query attention in the larger context of attention:

*   **Evolution of Attention:** The sources explain that the original Transformer architecture introduced **multi-head attention** to provide the model with "more extensive attention capability" by running the attention mechanism multiple times in parallel, allowing it to attend to different types of information simultaneously.

*   **Optimization Goal:** As Transformer models grew larger, the attention calculation became a computationally expensive part of the process. **Multi-query attention** was introduced as an "efficient attention tweak" to address this by **sharing the keys and values matrices between all attention heads**, thus reducing the size of the matrices involved and improving inference scalability.

*   **Grouped-Query as a Middle Ground:** **Grouped-query attention** is described as building upon multi-query attention. Instead of sharing the keys and values matrices across *all* heads (as in multi-query attention), it allows for using **multiple groups of shared key/value matrices**, where each group has its respective set of attention heads. This represents a **trade-off between the high efficiency of multi-query attention and the potentially better model quality achievable with more parameters** in standard multi-head attention. The sources note that grouped-query attention "sacrifices a little bit of the efficiency of multi-query attention in return for a large improvement in quality".

*   **Comparison to Other Attention Mechanisms:** Figure 3-25 visually compares the original multi-head attention, grouped-query attention, and multi-query attention, highlighting how the sharing of keys and values differs among these methods. In multi-head attention, each head has distinct query, key, and value matrices. Multi-query attention has distinct queries but shared keys and values across all heads. Grouped-query attention falls in between, with distinct queries but shared keys and values within groups of heads.

*   **Usage in Modern Models:** The sources explicitly mention that **grouped-query attention is used by models like Llama 2**. This indicates its relevance and adoption in contemporary large language models. Figure 3-30 also indicates that a "2024-era Transformer like Llama 3 features some tweaks like pre-normalization and an attention optimized with grouped-query attention and rotary embeddings".

*   **Impact on Transformer Blocks:** The attention layer, which can be implemented using grouped-query attention, is one of the two major components of a Transformer block, the other being a feedforward neural network. The attention layer is primarily responsible for incorporating relevant information from other input tokens.

In summary, these sources present grouped-query attention as a significant **optimization of the attention mechanism** that sits between the full parameterization of multi-head attention and the extreme parameter sharing of multi-query attention. It aims to provide a **better balance between computational efficiency during inference and model quality**, and its adoption in models like Llama 2 highlights its practical value in the field of large language models.

#### Local/sparse attention

These sources discuss **local/sparse attention** as one approach to creating **more efficient attention mechanisms** within the Transformer architecture. In the larger context of attention mechanisms, which are fundamental to the ability of Large Language Models (LLMs) to incorporate context while processing language, local/sparse attention aims to address the computational cost associated with the attention calculation, especially as models scale.

Here's what the sources say about local/sparse attention:

*   **Efficiency Improvement:** Local/sparse attention is presented as an idea to improve the **efficiency of the attention calculation**, which is highlighted as the most computationally expensive part of the Transformer process.

*   **Limiting Context:** The key idea behind local/sparse attention is to **limit the context of previous tokens that the model can attend to**. Instead of the **full attention** mechanism, where each token can attend to all previous tokens in the sequence, sparse attention restricts this scope. Figure 3-22 visually demonstrates this, showing that local attention "boosts performance by only paying attention to a small number of previous positions".

*   **Example in GPT-3:** The sources mention that **GPT-3 incorporates a mechanism of sparse attention**. However, it's crucial to note that GPT-3 does not use sparse attention for all its Transformer blocks. If the model could only see a limited number of previous tokens, the quality of the generated text would significantly degrade. Instead, the architecture of GPT-3 **interweaves full-attention and efficient-attention Transformer blocks**, alternating between them (e.g., blocks 1 and 3 might use full attention, while blocks 2 and 4 use sparse attention).

*   **Comparison with Full Attention:** Figure 3-23 provides a visual comparison between **full attention and sparse attention**. The figure uses color coding (explained in Figure 3-24) to show which previous tokens (light blue) can be attended to when processing the current token (dark blue). In full attention, all previous tokens can be attended to, whereas in sparse attention, only a subset of recent tokens is considered. The sources also point out in Figure 3-24 that decoder Transformer blocks (common in text generation models) can only attend to previous tokens, highlighting the autoregressive nature.

*   **Part of Efficient Attention Strategies:** Local/sparse attention is mentioned alongside **multi-query and grouped-query attention** as a "more recent efficient attention tweak to the Transformer". These methods, including sparse attention, aim to optimize the attention mechanism for various reasons, such as improving speed and reducing computational cost.

In the larger context of attention mechanisms, local/sparse attention represents a strategy to make Transformers more scalable and efficient, particularly for processing longer sequences. By selectively limiting the tokens that a given token attends to, it reduces the computational complexity compared to full self-attention. However, as highlighted by the example of GPT-3, relying solely on sparse attention can be detrimental to model performance, suggesting that a balance or a strategic interleaving with full attention mechanisms might be necessary in practice. The development of sparse attention underscores the ongoing research and innovation in optimizing the core attention mechanism of the Transformer to enable the training and deployment of increasingly large and capable language models.


#### Flash Attention

These sources discuss **Flash Attention** as a **recent and popular method and implementation that provides significant speedups for both training and inference of Transformer LLMs on GPUs**. It's presented in the larger context of the **attention layer of the Transformer**, which is identified as the **most computationally expensive part of the entire process**.

Here's a breakdown of what the sources say about Flash Attention:

*   **Optimization for Speed and Memory Efficiency:** Flash Attention is designed to **speed up the attention calculation** by **optimizing what values are loaded and moved between a GPU’s shared memory (SRAM) and high bandwidth memory (HBM)**. This optimization addresses a key bottleneck in Transformer performance.

*   **Context of Efficient Attention Techniques:** The discussion of Flash Attention appears within a section titled "**More Efficient Attention**" and is mentioned alongside other techniques like **local/sparse attention**, **multi-query attention**, and **grouped-query attention**. This places Flash Attention within the broader effort to make Transformer models more scalable and efficient, particularly for larger models.

*   **GPU Specific Optimization:** The sources explicitly state that Flash Attention provides speedups for LLMs on **GPUs**, indicating that it is a hardware-aware optimization. It leverages the specific memory architecture of GPUs to improve performance.

*   **Relationship to Multi-Head Attention:** While the sources don't directly compare Flash Attention to the *mechanism* of multi-head attention, the fact that it aims to speed up the *attention calculation* implies that it can be applied to multi-head attention as well. The fundamental concept of calculating queries, keys, and values across multiple heads remains, but Flash Attention optimizes the underlying computations and memory management on the GPU.

*   **Contrast with Architectural Tweaks:** Unlike **multi-query** and **grouped-query attention**, which involve architectural modifications to how keys and values are shared between attention heads to improve inference scalability by reducing matrix sizes, Flash Attention seems to be more of an **algorithmic and implementation-level optimization** focused on efficient computation and memory movement on GPUs.

*   **Further Research:** The sources mention the existence of research papers, specifically "**FlashAttention: Fast and memory-efficient exact attention with IO-awareness**" and "**FlashAttention-2: Faster attention with better parallelism and work partitioning**," which describe Flash Attention in detail. This highlights that Flash Attention is an active area of research and development aimed at further enhancing Transformer efficiency.

In summary, these sources present Flash Attention as a **critical advancement in optimizing the performance of Transformer models on GPUs**. It tackles the computational cost of the attention mechanism by intelligently managing data movement within the GPU's memory hierarchy. This places it within the larger context of ongoing efforts to make attention mechanisms, the core of Transformer models, more efficient, alongside architectural innovations like sparse, multi-query, and grouped-query attention.

---

