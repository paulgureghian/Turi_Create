"""Created in Mar 2019 by Paul A. Gureghian """

""" Implement Turi Create """ 

### Import packages
import turi_create as tc

### Train the TURI model
train_data, test_data = data.random_split(0.8) 

model = tc.object_detector.create(train_data, max_iterations=1000,
                                  feature='image', annotations='annotations')

### Visualize the data
test_data['predictions'] = model.predict(test_data)

test_data['image_with_predictions'] =
tc.object_detector.util.draw_bounding_boxes(test_data['image'],
test_data['predictions']) 

test_data.explore()

### Save the model
model.save('my.model') 

### Convert model to 'CoreML'  
model.export_coreml('my.model') 







 