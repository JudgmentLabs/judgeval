from judgeval.scorers import JudgevalScorer

class CustomScorer(JudgevalScorer):
    def __init__(
        self,
        threshold=0.5,
        score_type="Sample Scorer",
        include_reason=True,
        async_mode=True,
        strict_mode=False,
        verbose_mode=True,
        custom_example=True
    ):
        super().__init__(score_type=score_type, threshold=threshold, custom_example=custom_example)
        self.threshold = 1 if strict_mode else threshold
        # Optional attributes
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
    
    def score_example(self, example):
        try:
            self.score = 1
            self.success = True
        except Exception as e:
            self.error = str(e)
            self.success = False
            
    async def a_score_example(self, example):
        try:
            self.score = 1
            self.success = True
        except Exception as e:
            self.error = str(e)
            self.success = False

    def _success_check(self):
        return self.score >= self.threshold
    

    @property
    def __name__(self):
        return "Alan Scorer"