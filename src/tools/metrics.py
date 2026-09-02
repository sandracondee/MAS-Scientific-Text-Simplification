class MetricsEvaluator:

    def __init__(self):
        # Imports perezosos: evaluate y textstat son pesados y descargan
        # modelos. Se cargan solo la primera vez que se instancia, no al
        # importar el módulo, para que el arranque del contenedor en Render
        # sea ligero (plan free, 512 MB).
        import evaluate
        import textstat

        self._textstat = textstat

        print("Loading evaluation metrics: SARI, BLEU and FKGL.")
        self.sari_metric = evaluate.load("sari")
        self.bleu_metric = evaluate.load("bleu")
        print("Evaluation metrics loaded.")

    def calc_simplification_metrics(self, complex_text: str, current_simplified_text: str, reference_text: str) -> dict:
        """
        Calculates SARI, BLEU and FKGL metrics to evaluate the current simplified text.
        Receives the original text, the current simplified text given by the Plain Language Simplifier agent
        and the reference text.
        """
        sources = [complex_text]
        predictions = [current_simplified_text]
        references = [[reference_text]]

        try:
            sari_score = self.sari_metric.compute(
                sources=sources,
                predictions=predictions,
                references=references
            )
        except Exception as e:
            print(f"Error calculating SARI: {e}")


        try:
            bleu_score = self.bleu_metric.compute(
                predictions=predictions,
                references=references,
                smooth=True
            )
        except Exception as e:
            print(f"Error calculating BLEU: {e}")

        # FKGL
        fkgl_scores = [self._textstat.flesch_kincaid_grade(text) for text in predictions]

        return {
            "SARI": sari_score['sari'],
            "BLEU": bleu_score['bleu'],
            "FKGL": sum(fkgl_scores) / len(fkgl_scores)
        }
