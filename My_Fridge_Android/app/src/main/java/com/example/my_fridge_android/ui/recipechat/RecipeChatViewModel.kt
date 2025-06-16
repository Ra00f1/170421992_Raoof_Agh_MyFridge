package com.example.my_fridge_android.ui.recipechat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.my_fridge_android.domain.repository.MainRepository
import com.example.my_fridge_android.ui.recipechat.RecipeChatContract.UiAction
import com.example.my_fridge_android.ui.recipechat.RecipeChatContract.UiEffect
import com.example.my_fridge_android.ui.recipechat.RecipeChatContract.UiState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject
import org.json.JSONObject
import org.json.JSONArray
import org.json.JSONException

@HiltViewModel
class RecipeChatViewModel @Inject constructor(
    private val mainRepository: MainRepository
) : ViewModel() {

    companion object {
        private var pendingApiResponse: String? = null

        fun setApiResponse(response: String) {
            println("DEBUG: RecipeChatViewModel.Companion - Setting pending API response: $response")
            pendingApiResponse = response
        }
    }

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _uiEffect by lazy { Channel<UiEffect>() }
    val uiEffect: Flow<UiEffect> by lazy { _uiEffect.receiveAsFlow() }

    init {
        // Check if there's a pending API response
        println("DEBUG: RecipeChatViewModel.init - Checking for pending API response...")
        pendingApiResponse?.let { response ->
            println("DEBUG: RecipeChatViewModel.init - Found pending API response: $response")
            initializeWithApiResponse(response)
            pendingApiResponse = null // Clear after use
            println("DEBUG: RecipeChatViewModel.init - Cleared pending API response")
        } ?: run {
            println("DEBUG: RecipeChatViewModel.init - No pending API response, showing welcome message")
            // Add welcome message if no API response
            val welcomeMessage = ChatMessage(
                id = UUID.randomUUID().toString(),
                content = "Hi! I'm your recipe assistant. Tell me what you want to cook or what ingredients you have, and I'll suggest some delicious recipes for you! 🍳👨‍🍳",
                isFromUser = false
            )
            _uiState.update {
                it.copy(messages = listOf(welcomeMessage))
            }
        }
    }

    fun onAction(uiAction: UiAction) {
        when (uiAction) {
            is UiAction.MessageChanged -> {
                updateUiState {
                    copy(currentMessage = uiAction.message)
                }
            }

            is UiAction.SendMessage -> {
                sendMessage()
            }

            is UiAction.BackToHome -> {
                viewModelScope.launch {
                    emitUiEffect(UiEffect.NavigateToHome)
                }
            }

            is UiAction.ClearError -> {
                updateUiState {
                    copy(errorMessage = "")
                }
            }

            is UiAction.InitializeWithApiResponse -> {
                initializeWithApiResponse(uiAction.response)
            }

            is UiAction.SelectOption -> {
                selectOption(uiAction.optionNumber)
            }
        }
    }

    private fun initializeWithApiResponse(response: String) {
        // Debug logging
        println("DEBUG: RecipeChatViewModel - Received API response from /tarif_oner: $response")

        // Parse and beautify JSON response with Turkish text support
        val beautifiedResponse = beautifyJsonResponse(response)

        // Clear the welcome message and add the API response
        val responseMessage = ChatMessage(
            id = UUID.randomUUID().toString(),
            content = "🤖 Recipe Suggestions from your fridge items:\n\n$beautifiedResponse",
            isFromUser = false,
            hasOptions = true,
            options = listOf("Option 1", "Option 2", "Option 3")
        )

        println("DEBUG: RecipeChatViewModel - Displaying actual /tarif_oner response in chat")

        updateUiState {
            copy(
                messages = listOf(responseMessage),
                isInitialResponse = true,
                initialApiResponse = response,
                waitingForOptionSelection = true
            )
        }

        println("DEBUG: RecipeChatViewModel - Chat updated with actual API response content")
    }

    private fun beautifyJsonResponse(response: String): String {
        return try {
            val json = JSONObject(response)

            // Create a more readable format
            val builder = StringBuilder()

            // Parse common fields that might be in the response
            json.keys().forEach { key ->
                val value = json.get(key)

                when {
                    key.equals("recipes", ignoreCase = true) || key.equals(
                        "tarifler",
                        ignoreCase = true
                    ) -> {
                        builder.append("🍽️ **Recipes / Tarifler:**\n\n")
                        formatRecipes(value, builder)
                    }

                    key.equals("message", ignoreCase = true) || key.equals(
                        "mesaj",
                        ignoreCase = true
                    ) -> {
                        builder.append("💬 **Message:**\n")
                        builder.append("${value}\n\n")
                    }

                    key.equals("suggestions", ignoreCase = true) || key.equals(
                        "öneriler",
                        ignoreCase = true
                    ) -> {
                        builder.append("✨ **Suggestions:**\n")
                        formatSuggestions(value, builder)
                    }

                    key.equals("options", ignoreCase = true) || key.equals(
                        "seçenekler",
                        ignoreCase = true
                    ) -> {
                        builder.append("🔢 **Options:**\n")
                        formatOptions(value, builder)
                    }

                    else -> {
                        builder.append("📝 **${key.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }}:**\n")
                        builder.append("${value}\n\n")
                    }
                }
            }

            if (builder.isEmpty()) {
                // If no recognized fields, format as pretty JSON
                json.toString(4)
            } else {
                builder.toString().trim()
            }

        } catch (e: JSONException) {
            // If it's not valid JSON, check if it looks like a formatted response
            if (response.contains("{") || response.contains("[")) {
                // Try to format it as best as we can
                response.replace("\\n", "\n")
                    .replace("\\t", "    ")
                    .replace("\\/", "/")
            } else {
                // Plain text response
                response
            }
        }
    }

    private fun formatRecipes(value: Any, builder: StringBuilder) {
        try {
            when (value) {
                is JSONArray -> {
                    for (i in 0 until value.length()) {
                        val recipe = value.getJSONObject(i)
                        builder.append("${i + 1}. ")

                        if (recipe.has("name") || recipe.has("isim")) {
                            val name = recipe.optString("name", recipe.optString("isim", ""))
                            builder.append("**$name**\n")
                        }

                        if (recipe.has("ingredients") || recipe.has("malzemeler")) {
                            val ingredients = recipe.optJSONArray("ingredients")
                                ?: recipe.optJSONArray("malzemeler")
                            builder.append("   📋 Ingredients: ")
                            ingredients?.let { ing ->
                                for (j in 0 until ing.length()) {
                                    builder.append(ing.getString(j))
                                    if (j < ing.length() - 1) builder.append(", ")
                                }
                            }
                            builder.append("\n")
                        }

                        if (recipe.has("instructions") || recipe.has("tarif")) {
                            val instructions =
                                recipe.optString("instructions", recipe.optString("tarif", ""))
                            builder.append("   👨‍🍳 Instructions: $instructions\n")
                        }

                        builder.append("\n")
                    }
                }

                is JSONObject -> {
                    builder.append(value.toString(4))
                    builder.append("\n\n")
                }

                else -> {
                    builder.append("$value\n\n")
                }
            }
        } catch (e: Exception) {
            builder.append("$value\n\n")
        }
    }

    private fun formatSuggestions(value: Any, builder: StringBuilder) {
        try {
            when (value) {
                is JSONArray -> {
                    for (i in 0 until value.length()) {
                        builder.append("${i + 1}. ${value.getString(i)}\n")
                    }
                    builder.append("\n")
                }

                else -> {
                    builder.append("$value\n\n")
                }
            }
        } catch (e: Exception) {
            builder.append("$value\n\n")
        }
    }

    private fun formatOptions(value: Any, builder: StringBuilder) {
        try {
            when (value) {
                is JSONArray -> {
                    for (i in 0 until value.length()) {
                        builder.append("${i + 1}. ${value.getString(i)}\n")
                    }
                    builder.append("\n")
                }

                else -> {
                    builder.append("$value\n\n")
                }
            }
        } catch (e: Exception) {
            builder.append("$value\n\n")
        }
    }

    private fun selectOption(optionNumber: String) {
        val currentState = _uiState.value

        // Validate option selection (only 1, 2, or 3 allowed)
        if (optionNumber !in listOf("1", "2", "3")) {
            // Add error message
            val errorMessage = ChatMessage(
                id = UUID.randomUUID().toString(),
                content = "Please select a valid option: 1, 2, or 3",
                isFromUser = false
            )

            updateUiState {
                copy(messages = messages + errorMessage)
            }
            return
        }

        // Add user's option selection
        val userMessage = ChatMessage(
            id = UUID.randomUUID().toString(),
            content = optionNumber,
            isFromUser = true
        )

        // Add confirmation message
        val confirmationMessage = ChatMessage(
            id = UUID.randomUUID().toString(),
            content = "You selected option $optionNumber. Here's your detailed recipe:",
            isFromUser = false
        )

        updateUiState {
            copy(
                messages = messages + userMessage + confirmationMessage,
                waitingForOptionSelection = false,
                currentMessage = ""
            )
        }

        // Make API call with the selected option
        viewModelScope.launch {
            mainRepository.sendRecipeSelection(
                originalResponse = currentState.initialApiResponse,
                userSelection = optionNumber
            )
                .onSuccess { response ->
                    val beautifiedResponse = beautifyJsonResponse(response)
                    val recipeMessage = ChatMessage(
                        id = UUID.randomUUID().toString(),
                        content = "🍽️ **Detailed Recipe:**\n\n$beautifiedResponse",
                        isFromUser = false
                    )

                    updateUiState {
                        copy(
                            messages = messages + recipeMessage,
                            isChatDisabled = true // Disable chat after final response
                        )
                    }
                }
                .onFailure { error ->
                    val errorMessage = ChatMessage(
                        id = UUID.randomUUID().toString(),
                        content = "❌ Failed to get detailed recipe: ${error.message}",
                        isFromUser = false
                    )

                    updateUiState {
                        copy(messages = messages + errorMessage)
                    }
                    emitUiEffect(UiEffect.ShowError("Failed to get detailed recipe"))
                }
        }
    }

    private fun sendMessage() {
        val currentState = _uiState.value
        val message = currentState.currentMessage.trim()

        if (message.isBlank()) return

        // If waiting for option selection, treat this as option selection
        if (currentState.waitingForOptionSelection) {
            selectOption(message)
            return
        }

        // Regular message sending (existing functionality)
        val userMessage = ChatMessage(
            id = UUID.randomUUID().toString(),
            content = message,
            isFromUser = true
        )

        updateUiState {
            copy(
                messages = messages + userMessage,
                currentMessage = "",
                isMessageSending = true
            )
        }

        // Send to API and get response
        viewModelScope.launch {
            try {
                val recipes = getRecipeSuggestions(message)

                val botMessage = ChatMessage(
                    id = UUID.randomUUID().toString(),
                    content = "Here are some recipe suggestions for you:",
                    isFromUser = false,
                    recipes = recipes
                )

                updateUiState {
                    copy(
                        messages = messages + botMessage,
                        isMessageSending = false
                    )
                }

            } catch (e: Exception) {
                updateUiState {
                    copy(
                        isMessageSending = false,
                        errorMessage = "Failed to get recipe suggestions. Please try again."
                    )
                }
                emitUiEffect(UiEffect.ShowError("Failed to get recipe suggestions"))
            }
        }
    }

    private suspend fun getRecipeSuggestions(prompt: String): List<Recipe> {
        // Simulate API delay
        kotlinx.coroutines.delay(1500)

        // Mock recipes based on prompt - replace with actual API call
        return when {
            prompt.contains("pasta", ignoreCase = true) -> listOf(
                Recipe(
                    title = "Spaghetti Carbonara",
                    ingredients = listOf(
                        "400g spaghetti",
                        "200g bacon",
                        "4 eggs",
                        "100g Parmesan cheese",
                        "Black pepper",
                        "Salt"
                    ),
                    instructions = listOf(
                        "Cook spaghetti according to package instructions",
                        "Fry bacon until crispy",
                        "Beat eggs with Parmesan cheese",
                        "Mix hot pasta with bacon and egg mixture",
                        "Season with salt and pepper"
                    ),
                    cookingTime = "20 minutes",
                    difficulty = "Medium"
                ),
                Recipe(
                    title = "Penne Arrabbiata",
                    ingredients = listOf(
                        "400g penne",
                        "400g canned tomatoes",
                        "4 garlic cloves",
                        "2 dried chilies",
                        "Olive oil",
                        "Fresh basil"
                    ),
                    instructions = listOf(
                        "Cook penne according to package instructions",
                        "Sauté garlic and chilies in olive oil",
                        "Add tomatoes and simmer for 15 minutes",
                        "Toss with pasta and garnish with basil"
                    ),
                    cookingTime = "25 minutes",
                    difficulty = "Easy"
                )
            )

            prompt.contains("chicken", ignoreCase = true) -> listOf(
                Recipe(
                    title = "Chicken Teriyaki",
                    ingredients = listOf(
                        "4 chicken thighs",
                        "3 tbsp soy sauce",
                        "2 tbsp honey",
                        "1 tbsp rice vinegar",
                        "1 tsp ginger",
                        "2 cloves garlic"
                    ),
                    instructions = listOf(
                        "Mix soy sauce, honey, vinegar, ginger, and garlic",
                        "Marinate chicken for 30 minutes",
                        "Cook chicken in a pan until golden",
                        "Pour sauce over chicken and simmer until glazed"
                    ),
                    cookingTime = "45 minutes",
                    difficulty = "Easy"
                )
            )

            else -> listOf(
                Recipe(
                    title = "Quick Vegetable Stir Fry",
                    ingredients = listOf(
                        "Mixed vegetables",
                        "Soy sauce",
                        "Garlic",
                        "Ginger",
                        "Sesame oil",
                        "Rice"
                    ),
                    instructions = listOf(
                        "Heat oil in a wok or large pan",
                        "Add garlic and ginger, stir for 30 seconds",
                        "Add vegetables and stir-fry for 5-7 minutes",
                        "Season with soy sauce and serve over rice"
                    ),
                    cookingTime = "15 minutes",
                    difficulty = "Easy"
                ),
                Recipe(
                    title = "Classic Grilled Cheese",
                    ingredients = listOf("2 slices bread", "2 slices cheese", "Butter"),
                    instructions = listOf(
                        "Butter one side of each bread slice",
                        "Place cheese between bread, buttered sides out",
                        "Cook in pan until golden brown on both sides"
                    ),
                    cookingTime = "10 minutes",
                    difficulty = "Easy"
                )
            )
        }
    }

    private fun getDetailedRecipeForOption(option: Int): Recipe {
        return when (option) {
            1 -> Recipe(
                title = "Option 1 Recipe - Pasta Primavera",
                ingredients = listOf(
                    "400g pasta",
                    "2 cups mixed vegetables",
                    "1/2 cup heavy cream",
                    "1/4 cup parmesan cheese",
                    "2 cloves garlic",
                    "Olive oil",
                    "Salt and pepper"
                ),
                instructions = listOf(
                    "Cook pasta according to package directions",
                    "Sauté vegetables in olive oil until tender",
                    "Add garlic and cook for 1 minute",
                    "Stir in cream and parmesan",
                    "Toss with pasta and season"
                ),
                cookingTime = "20 minutes",
                difficulty = "Easy"
            )

            2 -> Recipe(
                title = "Option 2 Recipe - Chicken Stir Fry",
                ingredients = listOf(
                    "500g chicken breast",
                    "2 cups mixed vegetables",
                    "3 tbsp soy sauce",
                    "1 tbsp sesame oil",
                    "2 cloves garlic",
                    "1 tsp ginger",
                    "2 cups cooked rice"
                ),
                instructions = listOf(
                    "Cut chicken into strips",
                    "Heat oil in wok or large pan",
                    "Cook chicken until done, remove",
                    "Stir-fry vegetables with garlic and ginger",
                    "Return chicken, add sauce, serve over rice"
                ),
                cookingTime = "15 minutes",
                difficulty = "Medium"
            )

            3 -> Recipe(
                title = "Option 3 Recipe - Vegetable Soup",
                ingredients = listOf(
                    "4 cups vegetable broth",
                    "2 cups mixed vegetables",
                    "1 can diced tomatoes",
                    "1 onion, diced",
                    "2 cloves garlic",
                    "2 tbsp olive oil",
                    "Herbs and spices"
                ),
                instructions = listOf(
                    "Heat oil in large pot",
                    "Sauté onion and garlic until soft",
                    "Add vegetables and cook 5 minutes",
                    "Add broth and tomatoes",
                    "Simmer 20 minutes, season to taste"
                ),
                cookingTime = "30 minutes",
                difficulty = "Easy"
            )

            else -> Recipe(
                title = "Default Recipe",
                ingredients = listOf("Various ingredients"),
                instructions = listOf("Follow cooking steps"),
                cookingTime = "Varies",
                difficulty = "Easy"
            )
        }
    }

    private fun updateUiState(block: UiState.() -> UiState) {
        _uiState.update(block)
    }

    private suspend fun emitUiEffect(uiEffect: UiEffect) {
        _uiEffect.send(uiEffect)
    }
}
