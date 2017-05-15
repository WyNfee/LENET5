%This function will preview the image
%Param:
%p_training_data: the training data
%p_training_answer: the training answer
%p_image_label: the image label
%no output
function function_Preview_Image(p_training_data,p_training_answer, p_image_label)
    t_start_preview = true;
    t_size_data_amount = size(p_training_data, 1);
    t_size_image_data = size(p_training_data, 2);
     
    while(t_start_preview)
        %pick the index of the image
        t_pickedIndex= floor(rand(1) * t_size_data_amount);
        
        %extract the image data from data set
        t_training_image_data = p_training_data(t_pickedIndex, :);
        t_answer = p_training_answer(t_pickedIndex);
       
        t_image_matrix = reshape(t_training_image_data, sqrt(t_size_image_data), sqrt(t_size_image_data));

        imagesc(t_image_matrix);
        axis image off;
        title(['pre-processed data', p_image_label(t_answer)]);
        
        s = input('press enter to next preview, press q to end preview:','s');
        if(s == 'q')
            t_start_preview = false;
            fprintf('Preview End\n');
        end
        
    end

end