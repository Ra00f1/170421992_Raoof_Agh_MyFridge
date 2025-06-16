package com.example.my_fridge_android.ui.recipechat

import androidx.compose.ui.tooling.preview.PreviewParameterProvider
import com.example.my_fridge_android.ui.recipechat.RecipeChatContract.UiState
import java.util.UUID

class RecipeChatScreenPreviewProvider : PreviewParameterProvider<UiState> {

    override val values: Sequence<UiState> = sequenceOf(
        UiState(),
        UiState(
            messages = listOf(
                ChatMessage(
                    id = UUID.randomUUID().toString(),
                    content = "Hi! I'm your recipe assistant. Tell me what you want to cook or what ingredients you have, and I'll suggest some delicious recipes for you! 🍳👨‍🍳",
                    isFromUser = false
                ),
                ChatMessage(
                    id = UUID.randomUUID().toString(),
                    content = "I want to make pasta",
                    isFromUser = true
                ),
                ChatMessage(
                    id = UUID.randomUUID().toString(),
                    content = "Here are some recipe suggestions for you:",
                    isFromUser = false,
                    recipes = listOf(
                        Recipe(
                            title = "Spaghetti Carbonara",
                            ingredients = listOf(
                                "400g spaghetti",
                                "200g bacon",
                                "4 eggs",
                                "100g Parmesan cheese"
                            ),
                            instructions = listOf(
                                "Cook spaghetti",
                                "Fry bacon",
                                "Mix eggs with cheese",
                                "Combine all"
                            ),
                            cookingTime = "20 minutes",
                            difficulty = "Medium"
                        )
                    )
                )
            ),
            currentMessage = "What about chicken?"
        ),
        UiState(
            messages = listOf(
                ChatMessage(
                    id = UUID.randomUUID().toString(),
                    content = "Hi! I'm your recipe assistant.",
                    isFromUser = false
                )
            ),
            isMessageSending = true
        )
    )
}