%this function is convert the 1 dimension answer matrix to two dimension
%answer matrix
%this is necesary conversion because we are using softmax
%param:
%p_input_answer: the answer matrix with one dimension
%p_output_amount: the output amount of the whole learning network
%return
%r_answer_matrix:the answer matrix with two dimension, ready for learning algorithm to use
function r_answer_matrix = function_Build_Answer_Matrix(p_input_answer, p_output_amount)
        t_input_answer_amount = size(p_input_answer,1);
        t_input_answer_matrix = zeros(t_input_answer_amount, p_output_amount);
        for i = 1 : t_input_answer_amount
            t_input_answer_matrix(i,p_input_answer(i)) = 1;
        end
        r_answer_matrix = t_input_answer_matrix;
end