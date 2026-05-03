"""VLLM entailment model with configurable preprocessing and postprocessing."""

import asyncio
import json
import logging
from typing import List, Optional
from .base import EntailmentModel, EntailmentResult
from ..processors import Preprocessor, Postprocessor, StandardPreprocessor, StandardPostprocessor

logger = logging.getLogger(__name__)


# Entailment system prompt
UPHILL_ENTAILMENT_SYSTEM = """Using your best judgment, indicate the agreement between the claim and the paragraph based on the opinion they express.
The information in the given texts may be true or false in the real world.
Please answer based only on the meaning of the text and disregard any knowledge or assumptions you may have about the text.
The response should be a dictionary with three keys - "reasoning", "agreement" and "unsure" which correspond to the reasoning, whether the given paragraph agrees or disagrees with the claim or none of them (Agree or Disagree or Neutral), and if you are unsure about the agreement.
You should only respond in the JSON as described below. 

<response-format>
{ 
  "reasoning": "How does the given paragraph agree or disagree with the claim? Be careful when you say the paragraph agrees or disagrees with the claim. You must provide reasoning to support your decision.",
  "agreement": "agree" if the paragraph agrees with the claim or supports it. "disagree" if the paragraph disagrees with the claim or is skeptic about it. "neutral" if the paragraph neither agrees or disagrees with the claim, or both agrees and disagrees with it. 
  "unsure": If the text is hard to understand or you are unsure of the label, answer True. False otherwise. 
} 
</response-format>

Several examples are given below. 

<example>
Claim: Annual mammograms may have more false-positives 
Paragraph: Annual mammograms have been the recommended screening tool for breast cancer detection for decades. However, in recent years, some studies have suggested that annual mammograms may result in more false-positives than biennial mammograms, thereby leading to unnecessary anxiety, stress, and medical interventions. A false-positive mammogram is one that suggests the presence of breast cancer, but further testing, such as ultrasounds, MRI scans, or biopsies, reveal that no cancer is present. False-positives are a common occurrence in mammography, particularly in healthy women who do not have any risk factors for developing breast cancer. However, repeated false-positives can result in unnecessary anxiety, which may lead to overdiagnosis and overtreatment. While false-positives and overdiagnosis are possible risks associated with annual mammography, many experts still recommend annual screening for women who are at high risk of developing breast cancer. 

{ 
  "reasoning": "The paragraph agrees with the claim because the paragraph says that false-positives are a common occurrence in mammography and annual mammograms may result in more false-positives than biennial mammograms.",
  "agreement": "agree", 
  "unsure": false
} 
</example>

<example>
Claim: The mortality rate for the flu is right around that of the new coronavirus: It's right around 2%.
Paragraph: The mortality rate for the flu varies every year, but it usually ranges from 0.1% to 0.2%. However, the mortality rate for COVID-19 seems to be higher. According to a study published in The Lancet Infectious Diseases in March 2020, the global case-fatality rate (CFR) for COVID-19 was estimated to be 2.3%. However, this varies by age group and underlying health conditions. In another study published in the European Respiratory Journal in April 2020, the CFR for COVID-19 was found to be 1.4%. However, this study only looked at cases in Europe.

{ 
  "reasoning": "The paragraph disagrees with the claim as the mortality rate for flu (0.1% to 0.2%) is NOT right around that of coronavirus (1.4% to 2.3%).",
  "agreement": "disagree",
  "unsure": false
}
</example>
  
<example>
Claim: Study: Vaccine for Breast, Ovarian Cancer Has Potential
Paragraph: It is worth noting that studies on vaccines for breast and ovarian cancer are ongoing. In general, these studies involve the development and testing of vaccines that aim to trigger an immune response against cancer cells. Researchers hope that these vaccines will help prevent or treat these types of cancers in the future. Some promising approaches involve using proteins found on cancer cells to stimulate the immune system, or using genetically modified viruses to deliver cancer-fighting genes to the body. However, more research is needed before these vaccines can be widely available for clinical use.

{
  "reasoning": "The paragraph is neutral with respect to the claim as knowing that studies for the vaccine are ongoing and that the researchers are hopeful it will help prevent or treat cancers does not necessarily imply that these vaccines have potential to cure the disease.",
  "agreement": "neutral",
  "unsure": false 
}
</example>

Good luck!

# Response Formats
## evaluation_response
{"properties": {"reasoning": {"type": "string"}, "agreement": {"type": "string", "enum": ["agree", "disagree", "neutral"]}, "unsure": {"type": "boolean"}}, "required": ["reasoning", "agreement", "unsure"], "type": "object"}
"""


def create_entailment_message(claim: str, response: str) -> str:
    """Create entailment prompt message."""
    return f"""Claim: {claim}\nParagraph: {response}"""


class VLLMEntailmentModel(EntailmentModel):
    """Unified VLLM entailment model with configurable processors."""

    def __init__(
        self,
        model_path: str,
        preprocessor: Optional[Preprocessor] = None,
        postprocessor: Optional[Postprocessor] = None,
        **kwargs,
    ):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self.kwargs["model_path"] = model_path
        self.batch_size = kwargs.get("batch_size", 64)
        self.llm = None

        # Use standard processors by default
        self.preprocessor = preprocessor or StandardPreprocessor()
        self.postprocessor = postprocessor or StandardPostprocessor()

    async def load(self):
        """Load the model and initialize resources."""
        if self.is_loaded:
            return

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("VLLMEntailmentModel requires vllm package") from e

        logger.info(f"Loading vLLM entailment model: {self.model_path}")

        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        self.is_loaded = True
        logger.info(f"vLLM entailment model loaded successfully: {self.model_path}")

    def _load_model(self):
        """Synchronous model loading."""
        from vllm import LLM

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.kwargs.get("tensor_parallel_size", 1),
            data_parallel_size=self.kwargs.get("data_parallel_size", 1),
            pipeline_parallel_size=self.kwargs.get("pipeline_parallel_size", 1),
            enable_prefix_caching=self.kwargs.get("enable_prefix_caching", True),
            gpu_memory_utilization=self.kwargs.get("gpu_memory_utilization", 0.9),
            enable_expert_parallel=self.kwargs.get("enable_expert_parallel", False),
        )

    async def unload(self):
        """Unload the model and free resources."""
        if not self.is_loaded:
            return

        if self.llm is not None:
            logger.info(f"Unloading vLLM entailment model: {self.model_path}")
            del self.llm
            self.llm = None

        self.is_loaded = False
        logger.info(f"vLLM entailment model unloaded: {self.model_path}")

    async def check_entailment(
        self, claims: List[str], responses: List[str], **kwargs
    ) -> List[EntailmentResult]:
        """Check entailment using configurable processors with JSON extraction."""
        if not self.is_loaded or self.llm is None:
            raise RuntimeError("Model not loaded. Call load() before check_entailment()")

        # Merge kwargs
        merged_kwargs = self._merge_kwargs(**kwargs)

        # Create prompts
        prompts = [
            create_entailment_message(claim, response) for claim, response in zip(claims, responses)
        ]

        # Run evaluation in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._check_entailment_sync, prompts, merged_kwargs
        )

        return results

    def _check_entailment_sync(
        self, prompts: List[str], merged_kwargs: dict
    ) -> List[EntailmentResult]:
        """Synchronous entailment checking."""
        from vllm import SamplingParams

        # Add entailment system prompt for preprocessing
        entailment_kwargs = merged_kwargs.copy()
        entailment_kwargs["system_prompt"] = UPHILL_ENTAILMENT_SYSTEM

        # Preprocess prompts
        processed_inputs, updated_kwargs = self.preprocessor.process(prompts, **entailment_kwargs)

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=updated_kwargs.get("temperature", 1),
            max_tokens=updated_kwargs.get("max_tokens", 512),
            stop=updated_kwargs.get("stop", None),
            stop_token_ids=updated_kwargs.get("stop_token_ids", None),
        )

        # Generate responses
        if isinstance(processed_inputs[0], list) and isinstance(processed_inputs[0][0], dict):
            outputs = self.llm.chat(processed_inputs, sampling_params=sampling_params, use_tqdm=True)
        else:
            outputs = self.llm.generate(
                prompts=[{"prompt_token_ids": inp} for inp in processed_inputs],
                sampling_params=sampling_params,
                use_tqdm=True
            )

        # Parse results
        parsed_responses = []

        for output, prompt in zip(outputs, prompts):
            try:
                (response,) = self.postprocessor.process([output], [prompt])

                # Handle both string and tuple responses from postprocessor
                if isinstance(response, tuple):
                    response_text, _ = response
                    text_to_parse = response_text
                else:
                    text_to_parse = response

                # Clean the response (remove markdown, etc.)
                cleaned = text_to_parse.strip().lstrip("json").strip("```").strip()

                # Parse JSON
                data = json.loads(cleaned)
                result = EntailmentResult(
                    reasoning=data["reasoning"], entailment=data["agreement"], unsure=data["unsure"]
                )
                parsed_responses.append(result)
            except Exception as e:
                logger.error(f"Error parsing entailment response: {e}")
                # Return a default unsure response on parsing error
                parsed_responses.append(
                    EntailmentResult(
                        reasoning=f"Error parsing response: {e}",
                        entailment="error",
                        unsure=True,
                    )
                )

        return parsed_responses
