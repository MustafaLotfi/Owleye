Once I built an eye tracker using machine learning. Its performance was like this:
I collected dataset from 30 people. In this dataset, I saved images of eyes and head and the angles. These are for input. For output I saved the points that the subjects were looking. It was based on the monitor's screen. Then I build a CNN model using these input images and vectors, and output points. Now I have a base model.
After this step, for every other individual, I gathered fewer data to just calibrate the model on that person. With calibration I mean retrain or fine-tune the model on that specific subject.
The point was that when I fine-tuned the model on each subject, the output would converge on that person very much. I mean, the point of transfer learning is that the model will gain some knowledge about that task. But, when I retrained the model, it seemed that the model just care about the new subject that I was training the model on it. For retraining, I just made last layer trainable and the other were frozen. What was the problem?


ChatGPT said:
ChatGPT
Memory updated
The problem you’re describing is a classic case of overfitting during fine-tuning. When you fine-tuned your model on a specific subject, it became highly specialized to that individual’s data, causing it to lose its ability to generalize to other people. This issue is common when fine-tuning on small datasets, especially when the model adapts too strongly to the new subject-specific data.

Possible Reasons for Overfitting During Fine-Tuning:
Insufficient Data for Generalization: Fine-tuning on a small set of data from a new individual can cause the model to overfit that person’s specific features and behaviors, especially when there is not enough variation in the fine-tuning data.

Low Learning Rate but Insufficient Regularization: While a lower learning rate is generally used for fine-tuning, it might still not be enough to prevent the model from heavily adjusting to the small amount of new data. Without proper regularization techniques, the model can still overfit.

Fine-Tuning Only the Last Layer: Although you only retrained the last layer, this layer might still have enough capacity to overfit to the specific subject’s data, leading to poor generalization.

Model Bias Shift: The data from the new subject may have subtle biases or differences in eye movement patterns, lighting, or camera angle that are not representative of the original dataset. The model ends up learning these biases during fine-tuning.

Lack of Data Augmentation: If you didn’t use data augmentation during fine-tuning, the model may not have been exposed to enough variation, causing it to memorize specific characteristics of the new subject rather than learning more generalizable features.

Strategies to Improve Fine-Tuning and Maintain Generalization:
Use Regularization Techniques:

Apply dropout to the last few layers to reduce the chance of overfitting.
Implement weight decay (L2 regularization) to penalize overly confident weight adjustments.
Gradual Unfreezing of Layers:

Instead of fine-tuning only the last layer, gradually unfreeze more layers during fine-tuning. This approach allows the model to adjust intermediate features slowly, reducing the risk of overfitting.
Smaller Learning Rate with Early Stopping:

Use a very small learning rate and incorporate early stopping based on validation performance to prevent the model from overfitting on the calibration data.
Regularization by Mixing Original Data:

Mix a portion of the original dataset with the new subject’s data during fine-tuning. This helps retain the general knowledge learned during the initial training phase while adapting to the new subject.
Data Augmentation:

Apply data augmentation techniques like flipping, brightness adjustments, or slight rotations to artificially increase the diversity of the new subject’s data, which helps in reducing overfitting.
Use a Meta-Learning Approach:

Implement techniques like Model-Agnostic Meta-Learning (MAML) or similar approaches where the model is explicitly trained to quickly adapt to new data with minimal overfitting.
Calibrating Using Feature Extraction:

Instead of retraining the last layer, consider using the CNN as a feature extractor and calibrate using a lightweight model like a linear regression or a small neural network specifically tailored to adapt quickly to new subjects without disrupting the core learned features.
Adjust the Fine-Tuning Objective:

If available, use contrastive learning objectives that focus on maintaining consistency between the base model’s general knowledge and the fine-tuning process, ensuring that adaptation does not overly specialize the model.
Conclusion:
The key is to balance the adaptation to the new subject while preserving the general knowledge that allows the model to perform well across multiple subjects. By applying these techniques, you can refine the calibration process and achieve a model that is both subject-specific and generalizable.
