// ====================================================================
// UTILITY FUNCTIONS
// ====================================================================

// Survey form styling constants
const SURVEY_STYLES = {
    formContainer: 'style="text-align: left; max-width: 600px; margin: 0 auto;"',
    questionText: 'style="font-size: 0.9em; color: #666;"',
    questionSpacing: 'style="margin-top: 1.5em;"',
    optionContainer: 'style="margin: 0.5em 0;"',
    label: 'style="display: block; margin: 0.5em 0; cursor: pointer;"',
    inputRadio: 'style="margin-right: 8px;"',
    inputCheckbox: 'style="margin-right: 8px;"',
    textarea: 'style="width: 100%; font-family: inherit; font-size: 14px; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"',
    inputNumber: 'style="width: 200px; font-family: inherit; font-size: 14px; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"'
};

/**
 * Helper functions for building survey forms
 */
const SurveyHelpers = {
    /**
     * Creates a form container div
     */
    formContainer: function(content) {
        return '<div ' + SURVEY_STYLES.formContainer + '>' + content + '</div>';
    },
    
    /**
     * Creates a question with title and optional description
     */
    question: function(title, description) {
        let html = '<p><strong>' + title + '</strong></p>';
        if (description) {
            html += '<p ' + SURVEY_STYLES.questionText + '>' + description + '</p>';
        }
        return html;
    },
    
    /**
     * Creates a textarea input
     */
    textarea: function(name, rows, required) {
        const req = required ? ' required' : '';
        return '<textarea name="' + name + '" rows="' + rows + '" ' + SURVEY_STYLES.textarea + req + '></textarea>';
    },
    
    /**
     * Creates a number input
     */
    numberInput: function(name, min, max, required) {
        const req = required ? ' required' : '';
        return '<input type="number" name="' + name + '" min="' + min + '" max="' + max + '" ' + SURVEY_STYLES.inputNumber + req + '>';
    },
    
    /**
     * Creates a radio button option
     */
    radioOption: function(name, value, label, required) {
        const req = required ? ' required' : '';
        return '<label ' + SURVEY_STYLES.label + '><input type="radio" name="' + name + '" value="' + value + '" ' + SURVEY_STYLES.inputRadio + req + '> ' + label + '</label>';
    },
    
    /**
     * Creates a checkbox option
     */
    checkboxOption: function(name, value, label) {
        return '<label ' + SURVEY_STYLES.label + '><input type="checkbox" name="' + name + '" value="' + value + '" ' + SURVEY_STYLES.inputCheckbox + '> ' + label + '</label>';
    },
    
    /**
     * Creates a container for radio/checkbox options
     */
    optionsContainer: function(options) {
        return '<div ' + SURVEY_STYLES.optionContainer + '>' + options + '</div>';
    },
    
    /**
     * Creates a question with spacing
     */
    questionWithSpacing: function(title, description, content) {
        let html = '<p ' + SURVEY_STYLES.questionSpacing + '><strong>' + title + '</strong></p>';
        if (description) {
            html += '<p ' + SURVEY_STYLES.questionText + '>' + description + '</p>';
        }
        html += content;
        return html;
    }
};

/**
 * Helper function to validate forms and prevent navigation if invalid
 * @param {string} formId - ID of the form to validate
 * @param {Function} customValidation - Optional custom validation function
 * @returns {Function} - Function to be used as on_load callback
 */
function setupFormValidation(formId, customValidation) {
    return function() {
        // Use MutationObserver to catch button when it's added
        const observer = new MutationObserver(function(mutations) {
            const nextBtn = document.querySelector('#jspsych-instructions-next');
            const form = document.getElementById(formId);
            
            if (nextBtn && form) {
                // Add validation handler with capture phase (runs first)
                nextBtn.addEventListener('click', function(e) {
                    if (!validateForm(form, customValidation)) {
                        e.preventDefault();
                        e.stopPropagation();
                        e.stopImmediatePropagation();
                        return false;
                    }
                }, true); // Capture phase - runs before jsPsych's handler
                
                // Also prevent keyboard navigation
                const keyHandler = function(e) {
                    if (e.key === 'ArrowRight' || e.key === 'Enter') {
                        if (!validateForm(form, customValidation)) {
                            e.preventDefault();
                            e.stopPropagation();
                            return false;
                        }
                    }
                };
                document.addEventListener('keydown', keyHandler, true);
                
                // Stop observing once we've set up validation
                observer.disconnect();
            }
        });
        
        // Start observing
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // Also try immediately in case button already exists
        setTimeout(function() {
            const nextBtn = document.querySelector('#jspsych-instructions-next');
            const form = document.getElementById(formId);
            if (nextBtn && form) {
                nextBtn.addEventListener('click', function(e) {
                    if (!validateForm(form, customValidation)) {
                        e.preventDefault();
                        e.stopPropagation();
                        e.stopImmediatePropagation();
                        return false;
                    }
                }, true);
            }
        }, 50);
    };
}

/**
 * Validates a form using HTML5 validation and optional custom validation
 * @param {HTMLFormElement} form - The form element to validate
 * @param {Function} customValidation - Optional custom validation function
 * @returns {boolean} - True if form is valid, false otherwise
 */
function validateForm(form, customValidation) {
    if (!form) return true;
    
    // Check HTML5 validation
    if (!form.checkValidity()) {
        form.reportValidity();
        return false;
    }
    
    // Run custom validation if provided
    if (customValidation) {
        return customValidation(form);
    }
    
    return true;
}

