function medical_image_analysis_full()
    % Center the figure
    screenSize = get(0, 'ScreenSize');
    figWidth = 1200;
    figHeight = 720;
    figX = (screenSize(3) - figWidth) / 2;
    figY = (screenSize(4) - figHeight) / 2;

    fig = figure('Name', 'Medical Image Analysis', 'NumberTitle', 'off', ...
                 'Position', [figX, figY, figWidth, figHeight], ...
                 'Color', [0.95 0.95 1]);

    % Axes for image
    imgAxes = axes('Parent', fig, 'Units', 'pixels', ...
                   'Position', [230, 140, 650, 500]);
    title(imgAxes, 'Original Image', 'FontSize', 14, 'FontWeight', 'bold');

    % Buttons common settings
    buttonFontSize = 12;
    buttonColor = [0.8 0.9 1];

    % Buttons
    uicontrol('Style', 'pushbutton', 'String', 'Select Image', ...
              'Position', [40, 620, 150, 45], 'Callback', @selectImage, ...
              'FontSize', buttonFontSize, 'BackgroundColor', buttonColor, ...
              'TooltipString', 'Load an image to start analysis');

    uicontrol('Style', 'pushbutton', 'String', 'Grayscale', ...
              'Position', [40, 550, 150, 45], 'Callback', @grayscaleImage, ...
              'FontSize', buttonFontSize, 'BackgroundColor', buttonColor, ...
              'TooltipString', 'Convert image to grayscale');

    uicontrol('Style', 'pushbutton', 'String', 'Binarization', ...
              'Position', [40, 480, 150, 45], 'Callback', @binarizeImage, ...
              'FontSize', buttonFontSize, 'BackgroundColor', buttonColor, ...
              'TooltipString', 'Convert image to binary');

    uicontrol('Style', 'pushbutton', 'String', 'Noise Removal', ...
              'Position', [40, 410, 150, 45], 'Callback', @removeNoise, ...
              'FontSize', buttonFontSize, 'BackgroundColor', buttonColor, ...
              'TooltipString', 'Remove small noise particles');

    uicontrol('Style', 'pushbutton', 'String', 'Analyze Image', ...
              'Position', [40, 340, 150, 45], 'Callback', @analyzeImage, ...
              'FontSize', buttonFontSize, 'BackgroundColor', buttonColor, ...
              'TooltipString', 'Run full cancer analysis');

    % Result Panel
    resultPanel = uipanel('Title', 'Analysis Results', 'FontSize', 14, 'FontWeight', 'bold', ...
                          'Position', [0.78, 0.05, 0.2, 0.9], ...
                          'BackgroundColor', [0.9 0.95 1]);

    resultText = uicontrol('Parent', resultPanel, 'Style', 'edit', ...
                           'String', '', ...
                           'Units', 'normalized', ...
                           'Position', [0.05, 0.05, 0.9, 0.9], ...
                           'HorizontalAlignment', 'center', ...
                           'FontSize', 13, 'BackgroundColor', [0.9 0.95 1], ...
                           'Max', 10, 'Min', 0, 'Enable', 'inactive', ...
                           'ForegroundColor', [0 0 0]);

    % Status Text
    statusLabel = uicontrol('Style', 'text', 'String', '', ...
                           'Position', [230, 660, 650, 30], ...
                           'FontSize', 14, 'FontWeight', 'bold', ...
                           'BackgroundColor', [0.95 0.95 1], ...
                           'ForegroundColor', [0 0 0], ...
                           'HorizontalAlignment', 'center');

    % Load trained model
    load('pancreatic_cancer_CNN.mat', 'trainedNet');
    inputSize = trainedNet.Layers(1).InputSize;

    % Internal storage
    originalImg = [];
    analysisResult = '';

    %% Functions
    function selectImage(~, ~)
        [filename, pathname] = uigetfile({'.jpg;.png;.bmp;.tif'}, 'Select an Image');
        if isequal(filename, 0)
            return;
        end
        imagePath = fullfile(pathname, filename);
        originalImg = imread(imagePath);
        axes(imgAxes);
        imshow(originalImg);
        title(imgAxes, 'Original Image', 'FontSize', 14, 'FontWeight', 'bold');
        set(resultText, 'String', '');
        set(statusLabel, 'String', '');
    end

    function grayscaleImage(~, ~)
        if isempty(originalImg), return; end
        gray = rgb2gray(originalImg);
        axes(imgAxes);
        imshow(gray);
        title(imgAxes, 'Grayscale Image', 'FontSize', 14, 'FontWeight', 'bold');
    end

    function binarizeImage(~, ~)
        if isempty(originalImg), return; end
        gray = rgb2gray(originalImg);
        bin = imbinarize(gray);
        axes(imgAxes);
        imshow(bin);
        title(imgAxes, 'Binarized Image', 'FontSize', 14, 'FontWeight', 'bold');
    end

    function removeNoise(~, ~)
        if isempty(originalImg), return; end
        gray = rgb2gray(originalImg);
        bin = imbinarize(gray);
        cleaned = bwareaopen(bin, 100);
        axes(imgAxes);
        imshow(cleaned);
        title(imgAxes, 'Noise Removed Image', 'FontSize', 14, 'FontWeight', 'bold');
    end

    function analyzeImage(~, ~)
        if isempty(originalImg), return; end

        set(statusLabel, 'String', 'Analyzing... Please wait');
        pause(0.5); % Simulate loading time

        resizedImg = imresize(originalImg, inputSize(1:2));
        if size(resizedImg,3) == 1
            resizedImg = cat(3, resizedImg, resizedImg, resizedImg);
        end
        augimds = augmentedImageDatastore(inputSize(1:2), resizedImg);
        [predictedLabel, scores] = classify(trainedNet, augimds);

        cancerStatus = string(predictedLabel);
        confidence = max(scores) * 100;

        grayImg = rgb2gray(originalImg);
        bwImg = imbinarize(grayImg, 'adaptive');
        bwImg = bwareaopen(bwImg, 100);
        cc = bwconncomp(bwImg);
        wbcCount = cc.NumObjects;

        axes(imgAxes);
        imshow(originalImg);
        hold on;

        affectedCellCount = 0;
        if cancerStatus == "Present" || lower(cancerStatus) == "cancer"
            hsvImg = rgb2hsv(originalImg);
            h = hsvImg(:,:,1);
            s = hsvImg(:,:,2);
            v = hsvImg(:,:,3);

            purpleMask = (h >= 0.6 | h <= 0.1) & (s >= 0.4) & (v >= 0.2);
            purpleMask = imclose(purpleMask, strel('disk', 5));
            purpleMask = imopen(purpleMask, strel('disk', 3));
            purpleMask = bwareaopen(purpleMask, 50);

            stats = regionprops(purpleMask, 'BoundingBox');
            affectedCellCount = numel(stats);

            for i = 1:affectedCellCount
                rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
            end
        end
        hold off;

        % Result formatting
        if cancerStatus == "Present" || lower(cancerStatus) == "cancer"
            statusText = 'CANCER DETECTED';
            set(resultText, 'ForegroundColor', [1 0 0]);
        else
            statusText = 'NO CANCER DETECTED';
            set(resultText, 'ForegroundColor', [0 0.5 0]);
        end

        analysisResult = sprintf('White Blood Cell (WBC) Count: %d\nCancer Status: %s\nConfidence: %.2f%%\nAffected Cells: %d', ...
                                  wbcCount, statusText, confidence, affectedCellCount);

        set(resultText, 'String', analysisResult);
        title(imgAxes, sprintf('WBCs: %d | %s', wbcCount, statusText), 'FontSize', 14, 'FontWeight', 'bold');
        set(statusLabel, 'String', 'Analysis Complete');
    end
end   
[Uploading 221421601015,221421601021 final ppt.pptx…]()
[221421601015, 221421601021  Project Final Documention pdf.pdf](https://github.com/user-attachments/files/20036264/221421601015.221421601021.Project.Final.Documention.pdf.pdf)
[CONFERENCES CERTIFICATE.pdf](https://github.com/user-attachments/files/20036484/CONFERENCES.CERTIFICATE.pdf)
