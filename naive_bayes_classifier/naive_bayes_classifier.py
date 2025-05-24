from collections import defaultdict
from re import findall
from math import log

class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = defaultdict(int)  # Count of docs per class
        self.class_word_counts = defaultdict(lambda: defaultdict(int))  # Word counts per class
        self.class_total_words = defaultdict(int)  # Total words per class
        self.vocabulary = set()  # Unique words in training data
        self.classes = set()  # Unique classes

    def _tokenize(self, document):
        """Convert text to lowercase words (basic tokenization)"""
        return findall(r'\b\w+\b', document.lower())

    def fit(self, documents, classes):
        """Train the classifier with documents and their classes"""
        # Build vocabulary and count statistics
        for doc, cls in zip(documents, classes):
            self.classes.add(cls)
            self.class_counts[cls] += 1
            words = self._tokenize(doc)
            for word in words:
                self.vocabulary.add(word)
                self.class_word_counts[cls][word] += 1
                self.class_total_words[cls] += 1

    def _calculate_log_prior(self, cls):
        """Compute log prior probability for a class"""
        return log(self.class_counts[cls] / sum(self.class_counts.values()))

    def _calculate_log_likelihood(
        self, word, cls,
        alpha = 1
    ):
        """Compute smoothed log likelihood with Laplace smoothing"""
        count = self.class_word_counts[cls].get(word, 0) + alpha
        total = self.class_total_words[cls] + alpha * len(self.vocabulary)
        return log(count / total)

    def predict(self, document):
        """Predict class for a single document"""
        words = self._tokenize(document)
        max_log_prob = -float('inf')
        best_class = None
        for cls in self.classes:
            log_prob = self._calculate_log_prior(cls)
            for word in words:
                if word in self.vocabulary:  # Ignore unknown words
                    log_prob += self._calculate_log_likelihood(word, cls)
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = cls
        return best_class

    def evaluate(self, test_documents, test_classes):
        """Calculate accuracy on test data"""
        correct = 0
        for doc, true_cls in zip(test_documents, test_classes):
            pred_cls = self.predict(doc)
            if pred_cls == true_cls:
                correct += 1
        return correct / len(test_documents)

if __name__ == "__main__":
    train_docs = [
        "A great movie with fantastic acting",
        "Horrible plot and bad editing",
        "Wonderful cinematography and score",
        "Terrible dialogue and pacing",
        "Brilliant performance by lead actor"
    ]
    train_classes = ["positive", "negative", "positive", "negative", "positive"]
    test_docs = [
        "Amazing special effects and plot",
        "Worst movie I've ever seen"
    ]
    test_classes = ["positive", "negative"]
    # Create and train classifier
    nb = NaiveBayesClassifier()
    nb.fit(train_docs, train_classes)
    print("Test predictions:")
    for doc in test_docs:
        print(f"'{doc}' => {nb.predict(doc)}")
    accuracy = nb.evaluate(test_docs, test_classes)
    print(f"\nAccuracy: {accuracy:.1%}")
