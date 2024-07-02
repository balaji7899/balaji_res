clc
clear variables
close all

% eyeImages = cell(61, 1);
% % eyeLabels = randi([0, 1], numSamples, 1);
% eyeLabels = [0,0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,...
%     1,1,1,1,1,1,1,0,0,0,1,0,1,0,1,1,0,1,0,0,1,1,1,...
%     1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,1,0,1,1]';
% 
% folderPath = 'D:\NIT_SIP_All_files_&_Folders\MINI_PRO_MI\Eye_Project\eyes';  % Update with your image folder path
% 
% % Check if the folder exists
% if ~isfolder(folderPath)
%     error('Folder does not exist: %s', folderPath);
% end
% 
% % Get a list of all image files in the folder
% imageFiles = dir(fullfile(folderPath, '*.jpg')); 
% 
% for i = 1:numel(imageFiles)
%     % Create synthetic eye image (random noise for simplicity)
%     filename = fullfile(folderPath, imageFiles(i).name);
% %     eyeImage = rand(imageSize);
%     A= imread(filename);
% %     A=imread("eyes\AL2.JPG");
% %     figure
% %     imshow(A);
% 
%     image = A;
% 
%     % Convert the image to the Lab color space
%     lab_image = rgb2lab(image);
% 
%     % Reshape the image into feature vectors
%     [nrows, ncols, ~] = size(lab_image);
%     feature_vectors = reshape(lab_image, nrows*ncols, 3);
% 
%     % Perform k-means clustering
%     num_clusters = 2; % Adjust as needed
%     [cluster_indices, cluster_centroids] = kmeans(double(feature_vectors), num_clusters);
% 
%     % Select the cluster corresponding to the conjunctiva region
%     % You may choose the cluster based on its centroid or other characteristics
%     % For simplicity, we'll assume the cluster with the highest mean L value corresponds to the conjunctiva
%     [~, conjunctiva_cluster_index] = max(cluster_centroids(:,1));
% 
%     % Generate a binary mask indicating the conjunctiva region
%     conjunctiva_mask = reshape(cluster_indices == conjunctiva_cluster_index, nrows, ncols);
% 
%     % Apply morphological operations to refine the segmented conjunctiva region
%     se = strel('disk', 5);
%     segmented_image = imopen(conjunctiva_mask, se);
% 
%     % Convert the segmented region back to RGB
%     segmented_rgb = repmat(segmented_image, [1, 1, 3]);
% 
%     % Overlay the segmented region on the original image
%     segmented_conjunctiva = image;
%     segmented_conjunctiva(~segmented_rgb) = 0; % Set non-segmented regions to black
% 
%     % Display the segmented conjunctiva region
% %     figure
% %     imshow(segmented_conjunctiva);
% %     title('Segmented Conjunctiva Region');
%     normalizedImg = imresize(segmented_conjunctiva, [64, 64]);
%     
%     % Store image in cell array
%     eyeImages{i} = normalizedImg;
% end
% 
% % Define paths for saving images and labels
% imagesPath = 'D:\NIT_SIP_All_files_&_Folders\MINI_PRO_MI\Eye_Project\PP_eye_imgs'; % Update with your desired path
% labelsPath = 'D:\NIT_SIP_All_files_&_Folders\MINI_PRO_MI\Eye_Project\PP_eyes_labels'; % Update with your desired path
% 
% % Create directories if they don't exist
% if ~exist(imagesPath, 'dir')
%     mkdir(imagesPath);
% end
% if ~exist(labelsPath, 'dir')
%     mkdir(labelsPath);
% end
% 
% % Loop through images and save them with corresponding labels
% for i = 1:61
%     % Save image
%     img = eyeImages{i};
%     imgFilename = fullfile(imagesPath, sprintf('%05d.png', i)); % Adjust filename format as needed
%     imwrite(img, imgFilename);
%     
%     % Save label
%     labelFilename = fullfile(labelsPath, sprintf('%05d.txt', i)); % Adjust filename format as needed
%     label = num2str(eyeLabels(i));
%     fileID = fopen(labelFilename, 'w');
%     fprintf(fileID, label);
%     fclose(fileID);
% end


originalFolderPath = 'D:\NIT_SIP_All_files_&_Folders\MINI_PRO_MI\Eye_Project\PP_eye_imgs';  % Path to original segmented images
outputFolderPath = 'D:\NIT_SIP_All_files_&_Folders\MINI_PRO_MI\Eye_Project\img_set1';  % Path to output synthetic images
outputLabelPath = 'D:\NIT_SIP_All_files_&_Folders\MINI_PRO_MI\Eye_Project\label_set1';  % Path to output synthetic labels

% Check if the folders exist
if ~isfolder(originalFolderPath) || ~isfolder(outputFolderPath) || ~isfolder(outputLabelPath)
    error('One or more folders do not exist.');
end

% Define the number of synthetic images you want to create for each original image
syntheticPerImage = 80;

% Initialize a counter for synthetic images
counter = 1;

originalFiles = dir(fullfile(originalFolderPath, '*.png'));

for fileIdx = 1:numel(originalFiles)
    originalImageFilename = fullfile(originalFolderPath, originalFiles(fileIdx).name);
    originalImg = imread(originalImageFilename);   
    
    % Generate synthetic images by randomly changing hue
    for i = 1:syntheticPerImage
        % Process the image to create synthetic variations (random hue adjustment)
        hueOffset = randi([-30, 30]);  % Random hue offset within a range (-30 to 30)
        syntheticImg = originalImg;  % Copy the original image
        
        % Check if the image has a red channel (assuming RGB format)
       % Check if the image has a red channel (assuming RGB format)
% Check if the image has a red channel (assuming RGB format)
if size(syntheticImg, 3) == 3
    % Extract the red channel
    redChannel = syntheticImg(:, :, 1);
    
    % Define a threshold to identify red pixels (adjust as needed)
    redThreshold = 50;  % Example threshold for red channel intensity
    
    % Identify red pixels based on the threshold
    redPixels = redChannel > redThreshold;
    
    % Check if any red pixels are detected
    if any(redPixels(:))
%         disp('Red pixels found.');
        % Convert RGB to HSV for hue adjustment
        hsvImg = rgb2hsv(syntheticImg);
        
        % Iterate over red pixels and adjust hue individually
        for i = 1:size(hsvImg, 1)
            for j = 1:size(hsvImg, 2)
                if redPixels(i, j)
                    % Adjust hue for red pixel at (i, j)
                    hsvImg(i, j, 1) = mod(hsvImg(i, j, 1) + hueOffset / 360, 1);
                end
            end
        end
        
        % Convert back to RGB
        syntheticImg = hsv2rgb(hsvImg);
    else
        disp('No red pixels detected in the image.');
    end
else
    disp('Image does not have a red channel.');
end
        
        % Save the synthetic image
        outputImgFilename = fullfile(outputFolderPath, sprintf('synthetic_image%d.png', counter));  % Adjust output filename
        imwrite(syntheticImg, outputImgFilename);
        
        % Save the corresponding label (assuming you have labels)
        originalLabel = read_label_from_original(originalFiles(fileIdx));  % Implement this function to read labels
        outputLabelFilename = fullfile(outputLabelPath, sprintf('synthetic_image%d.txt', counter));  % Adjust output filename
        fileID = fopen(outputLabelFilename, 'w');
        fprintf(fileID, '%d', originalLabel);  % Assign the original label to synthetic image
        fclose(fileID);
        
        counter = counter + 1;
    end
end








% 
% DatasetPath=fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
% imds=imageDatastore(DatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
% Trainingdata = 700; %Spliting of data for training and testing
% [imdsTrain,imdsTest]=splitEachLabel(imds,Trainingdata,'randomize');
% c = countEachLabel(imds);
% totalImage = sum(table2array(c(:,2)));
% 
% Define the paths to your image and label folders
imageFolderPath = 'D:\NIT_SIP_All_files_&_Folders\MINI_PRO_MI\Eye_Project\img_set1';  % Update with your image folder path
labelFolderPath = 'D:\NIT_SIP_All_files_&_Folders\MINI_PRO_MI\Eye_Project\label_set1';  % Update with your label folder path

% Create an imageDatastore for images and specify the label source as folders
imds = imageDatastore(imageFolderPath, 'FileExtensions', {'.png', '.jpg', '.jpeg'}, 'LabelSource', 'foldernames');

% Create a cell array to hold the labels corresponding to each image
labels = cell(numel(imds.Files), 1);

% Loop through each image file to extract the corresponding label
for i = 1:numel(imds.Files)
    [~, imageName, ~] = fileparts(imds.Files{i});    
    labelFilename = fullfile(labelFolderPath, [imageName, '.txt']);  % Assuming label files have .txt extension
    
    % Read the label file (assuming it's a text file containing the label)
    if exist(labelFilename, 'file')
        fileID = fopen(labelFilename, 'r');
        label = fscanf(fileID, '%d');  % Read label as integer
        fclose(fileID);
        
        % Store the label in the labels cell array
        labels{i} = label;
    else
        error('Label file not found for image: %s', imds.Files{i});
    end
end

% Assign labels to the imageDatastore
imds.Labels = categorical(cell2mat(labels));

% Split the data into training and testing sets
Trainingdata = 1100; % Number of samples for training
[imdsTrain, imdsTest] = splitEachLabel(imds, Trainingdata, 'randomize');

% Check the count of each label in the training set
c = countEachLabel(imdsTrain);
totalImage = sum(c.Count);  % Total number of images in the training set



layers = [ % defining layers
 imageInputLayer([64 64 3])
 convolution2dLayer(3,8,'Padding',1)
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2)
 convolution2dLayer(3,16,'Padding',1)
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2)
 convolution2dLayer(3,32,'Padding',1)
 batchNormalizationLayer
 reluLayer
 fullyConnectedLayer(2)
 softmaxLayer
 classificationLayer];
figure;
% plot(layerGraph(layers));
title("Total number of Layers used for Training the Data");
options = trainingOptions('sgdm',... % Training options
 'MaxEpochs',3,...
 'InitialLearnRate',1e-1, ...
 'Verbose',false, ...
 'Plots','training-progress', ...
 'ValidationData',imdsTest);
% classifier = trainNetwork(imdsTrain,layers,options); %Training
% YPred = classify(classifier,imdsTest); % Predicting
YTest = imdsTest.Labels;

[net, traininfo] = trainNetwork(imdsTrain,layers,options);
[YPred, Probability] = classify(net,imdsTest);
close all
randomPermutation = randperm(0.3*totalImage,10);
figure
for i = 1:length(randomPermutation)
    subplot(5,2,i);
imshow(imdsTest.Files{randomPermutation(i)});
title(YPred(randomPermutation(i)));
end
sgtitle('Some Classified Images');

% Performance
accuracy = sum(YPred == YTest)/numel(YTest);
cm=confusionmat(YTest,YPred); %confusion matrix
figure;
plotconfusion(YTest,YPred);
cm=cm';
precision=diag(cm)./sum(cm,2); % precision of each class
overall_precision=mean(precision);
recall=diag(cm)./sum(cm,1)';
overall_recall=mean(recall);
f1score=(2*overall_precision*overall_recall)/(overall_precision+overall_recall);

