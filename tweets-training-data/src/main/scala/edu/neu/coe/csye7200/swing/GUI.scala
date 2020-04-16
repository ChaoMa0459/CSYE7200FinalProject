package edu.neu.coe.csye7200.swing

import javax.swing.ImageIcon

import scala.swing._
import scala.swing.event._
import edu.neu.coe.csye7200.unittests.Classification_models.build_model

object GUI extends SimpleSwingApplication {

  def top = new MainFrame {
    title = "Classification of Disaster Tweets"
    var tweet = new TextField()
    val font_1 = new Font("Ariel", java.awt.Font.BOLD, 18)
    val font_2 = new Font("Ariel", java.awt.Font.BOLD, 16)
    val font_3 = new Font("Ariel", java.awt.Font.BOLD, 14)
    var label1 = new Label {
      text = ("Enter the tweet:")
      font = font_1
    }
    var label2 = new Label("Num of trees:")
    var label3 = new Label("Max depth of trees:")
    var label4 = new Label("Random Seed:")
    var result = new Label {
      text = ("Result:")
      font = font_1
    }
    var time = new Label {
      text = ("Time Consumed:")
      font = font_1
    }
    var accuracy = new Label {
      text = ("Accuracy:")
      font = font_1
    }
    var button = new Button {
      text = "Start building your model and testing your tweet"
      font = font_3
    }
    var label5 = new Label {
      text = ("Please choose a model first:")
      font = font_1
    }
    var label6 = new Label {
      text = ("Set hyperparameters of model:")
      font = font_1
    }
    var label7 = new Label {
      text = ("Random Forest Classifier:")
      font = font_2
    }
    var label8 = new Label {
      text = ("Naive Bayesian Classifier:")
      font = font_2
    }
    var label9 = new Label("Smoothing:")
    var label10 = new Label {
      text = ("Linear Support Vector Classification:")
      font = font_2
    }
    var label11 = new Label("Max iterations:")
    var label12 = new Label("Regularization parameter:")
    var label13 = new Label("Standardize features:")
    var label14 = new Label("Fit an intercept term:")
    var image1 = new Label()
    {
      icon = new ImageIcon("src/main/resources/real_word_cloud.png")
    }
    var image2 = new Label()
    {
      icon = new ImageIcon("src/main/resources/fake_word_cloud.png")
    }

    var line_1 = new Separator
    var line_2 = new Separator
    var line_3 = new Separator
    val model1 = new RadioButton("Random Forest Classifier")
    val model2 = new RadioButton("Naive Bayesian Classifier")
    val model3 = new RadioButton("Linear Support Vector Classification")
    var R1 = new ComboBox(List(5, 10, 20, 30, 40, 50))
    var R2 = new ComboBox(List(5, 10, 15, 20, 25, 30))
    var R3 = new ComboBox(List(0, 5, 10, 20, 30, 40, 50))
    var N1 = new ComboBox(List(0, 1, 5, 10))
    var L1 = new ComboBox(List(2, 5, 20, 50, 100, 150))
    var L2 = new ComboBox(List(0.0, 0.1, 0.5, 1.0, 2.0))
    var L3 = new ComboBox(List(true, false))
    var L4 = new ComboBox(List(true, false))

    def left() ={
      for (e <- contents)
        e.xLayoutAlignment = 0.0
    }

    var algorithms = new ButtonGroup {
      buttons += model1
      buttons += model2
      buttons += model3
      model2.selected = true
    }

    def restrictHeight(s: Component) {
      s.maximumSize = new Dimension(Short.MaxValue, s.preferredSize.height)
    }

    contents = new BoxPanel(Orientation.Vertical) {
      contents += new GridPanel(1, 9) {
        contents += label1
        contents += Swing.HGlue
        contents += Swing.HGlue
        contents += Swing.HGlue
        contents += Swing.HGlue
        contents += Swing.HGlue
        contents += Swing.HGlue
        contents += Swing.HGlue
        contents += Swing.HGlue
      }
      contents += Swing.VStrut(10)
      contents += tweet

      contents += new Separator

      contents += new GridPanel(1, 3) {
        contents += new BoxPanel(Orientation.Vertical) {
          contents += label5
          contents += Swing.VStrut(10)
          contents ++= algorithms.buttons
        }
        contents += Swing.HGlue
        contents += Swing.HGlue
      }

      contents += new Separator

      contents += new GridPanel(1, 5) {
        contents += label6
        contents += Swing.HGlue
        contents += Swing.HGlue
        contents += Swing.HGlue
        contents += Swing.HGlue
      }

      contents += new Separator

      contents += new GridPanel(1, 3) {
        contents += new BoxPanel(Orientation.Vertical) {
          contents += label7
          contents += Swing.VStrut(10)
          contents += new BoxPanel(Orientation.Horizontal) {
            contents += label2
            contents += R1
          }
          contents += new BoxPanel(Orientation.Horizontal) {
            contents += label3
            contents += R2
          }
          contents += new BoxPanel(Orientation.Horizontal) {
            contents += label4
            contents += R3
          }
        }
        contents += new BoxPanel(Orientation.Vertical) {
          contents += label8
          contents += Swing.VStrut(10)
          contents += new BoxPanel(Orientation.Horizontal) {
            contents += label9
            contents += N1
          }
        }
        contents += new BoxPanel(Orientation.Vertical) {
          contents += label10
          contents += Swing.VStrut(10)
          contents += new BoxPanel(Orientation.Horizontal) {
            contents += label11
            contents += L1
          }
          contents += new BoxPanel(Orientation.Horizontal) {
            contents += label12
            contents += L2
          }
          contents += new BoxPanel(Orientation.Horizontal) {
            contents += label13
            contents += L3
          }
          contents += new BoxPanel(Orientation.Horizontal) {
            contents += label14
            contents += L4
          }
        }
      }
      contents += new Separator
      contents += Swing.VStrut(10)
      contents += Swing.VStrut(10)
      contents += new GridPanel(1, 3) {
        contents += Swing.HGlue
        contents += button
        contents += Swing.HGlue
      }
      contents += Swing.VStrut(10)
      contents += Swing.VStrut(10)
      contents += new Separator
      contents += new GridPanel(1, 2) {
        contents += new BoxPanel(Orientation.Vertical) {
          contents += result
          contents += accuracy
          contents += time
        }
        contents += Swing.HGlue
      }

      contents += new ScrollPane(new GridPanel(1, 2) {
        contents += image1
        contents += image2
      })
            restrictHeight(tweet)
    }
    var clicks = 0
    listenTo(button)
    reactions += {
      case ButtonClicked(b) if b == button =>
        clicks += 1
        val model = algorithms.selected.get.text.toString
        val text = tweet.text.toString
        val num_of_tree = R1.selection.item.toString
        val max_depth = R2.selection.item.toString
        val seed = R3.selection.item.toString
        val smoothing = N1.selection.item.toString
        val max_iter = L1.selection.item.toString
        val regularization = L2.selection.item.toString
        val standardize = L3.selection.item.toString
        val fit = L4.selection.item.toString
        if(text =="")
          {
            result.text = "Please enter a tweet!"
          }
        else
        {
          val result_data = build_model(text, model, num_of_tree, max_depth, seed, smoothing, max_iter, regularization, standardize, fit)
          val accuracy_data  = result_data._1
          val prediction = result_data._2
          result.text = "Result: "+ prediction
          accuracy.text = "Accuracy of "+model+" is "+accuracy_data.toString()
          time.text = "Time Consumed: "+(result_data._3 * 0.001).toString()+"s"
        }

    }
  }
}
