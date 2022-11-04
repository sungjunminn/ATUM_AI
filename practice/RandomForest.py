  #분류 코드
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

classifier = RandomForestClassifier(n_estimators = 100)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("정확도 : {}".format(accuracy_score(y_test, y_pred)))


#회귀 코드
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

regressor = RandomForestRegressor()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = np.sqrt(mean_squared_error(y_pred, y_test))

print('평균제곱근오차 : ', mse)


#GridSearchCV를 통한 랜덤포레스트분류의 하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV

grid = {
    'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,7,10],
    'min_samples_split' : [2,3,5,10]
}

classifier_grid = GridSearchCV(classifier, param_grid = grid, scoring="accuracy", n_jobs=-1, verbose =1)

classifier_grid.fit(X_train, y_train)

print("최고 평균 정확도 : {}".format(classifier_grid.best_score_))
print("최고의 파라미터 :", classifier_grid.best_params_)


#파라미터 중요도 시각화
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

feature_importances = model.feature_importances_

ft_importances = pd.Series(feature_importances, index = X_train.columns)
ft_importances = ft_importances.sort_values(ascending=False)

plt.figure(fig.size(12,10))
sns.barplot(x=ft_importances, y= X_train.columns)
plt.show()