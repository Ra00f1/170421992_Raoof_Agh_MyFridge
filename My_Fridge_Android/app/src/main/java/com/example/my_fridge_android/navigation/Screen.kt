package com.example.my_fridge_android.navigation

import kotlinx.serialization.Serializable

sealed interface Screen {
    @Serializable
    data object Login : Screen

    @Serializable
    data object Registe : Screen

    @Serializable
    data object Home : Screen

    @Serializable
    data object RecieptList : Screen

    @Serializable
    data object Profile : Screen

    @Serializable
    data object RecipeChat : Screen
}
